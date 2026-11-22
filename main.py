import io
import os
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import pandas as pd
import pdfplumber
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

app = FastAPI(title="Kevsight")

FEE_KEYWORDS = [
    "fee", "charge", "vat", "sms alert", "maintenance", "commission",
    "stamp duty", "levy", "penalty", "overdraft", "transfer fee", "service charge"
]

CATEGORY_RULES = {
    "Fees & Bank Charges": FEE_KEYWORDS,
    "Software & Subscriptions": [
        "aws", "google", "canva", "zoom", "slack", "notion", "openai",
        "anthropic", "github", "vercel", "render", "subscription"
    ],
    "Utilities & Internet": [
        "electric", "power", "water", "internet", "fiber", "data", "airtime", "utility"
    ],
    "Rent & Operations": ["rent", "lease", "office", "workspace", "storage"],
    "Payroll & Contractors": ["salary", "payroll", "wages", "contractor", "retainer", "payout"],
    "Sales & Revenue": [
        "invoice", "payment received", "settlement", "deposit", "sales",
        "stripe", "paystack", "flutterwave", "client", "order"
    ],
    "Supplies & Inventory": ["inventory", "stock", "supplies", "wholesale", "vendor", "supplier"]
}

RECENT_HISTORY: List[Dict[str, Any]] = []
MAX_HISTORY = 20


class AnalysisOptions(BaseModel):
    currency: str = "$"
    check_duplicates: bool = True


def categorize_description(desc: str) -> str:
    text = (desc or "").lower()
    for category, keywords in CATEGORY_RULES.items():
        if any(kw in text for kw in keywords):
            return category
    return "Other"


def parse_amount(value: Any) -> Optional[float]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    cleaned = str(value).strip()
    if not cleaned or cleaned in {"-", "--", "N/A", "n/a"}:
        return None
    negative = False
    if cleaned.startswith("(") and cleaned.endswith(")"):
        negative = True
        cleaned = cleaned[1:-1]
    cleaned = re.sub(r"[^\d.\-]", "", cleaned)
    if not cleaned or cleaned in {".", "-"}:
        return None
    try:
        num = float(cleaned)
        if negative and num > 0:
            num = -num
        return num
    except ValueError:
        return None


def normalize_dataframe(df: pd.DataFrame) -> List[Dict[str, Any]]:
    if df is None or df.empty:
        return []

    df = df.dropna(how="all").copy()
    df.columns = [str(c).strip().lower() for c in df.columns]

    date_col = next(
        (c for c in df.columns if any(k in c for k in ["date", "time", "posted", "value", "order_date"])),
        None
    )
    desc_col = next(
        (c for c in df.columns if any(k in c for k in [
            "desc", "narration", "details", "memo", "particulars",
            "merchant", "name", "item", "product", "customer"
        ])),
        None
    )
    amount_col = next(
        (c for c in df.columns if c in {"amount", "amt", "value", "total", "sales"} or "amount" in c or "total" in c),
        None
    )
    debit_col = next((c for c in df.columns if any(k in c for k in ["debit", "withdrawal", "outflow", "dr"])), None)
    credit_col = next((c for c in df.columns if any(k in c for k in ["credit", "deposit", "inflow", "cr"])), None)
    qty_col = next((c for c in df.columns if c in {"qty", "quantity", "units"}), None)
    price_col = next((c for c in df.columns if any(k in c for k in ["unit_price", "price", "rate"])), None)

    if not desc_col:
        non_numeric_cols = [c for c in df.columns if c != date_col]
        desc_col = non_numeric_cols[0] if non_numeric_cols else df.columns[0]

    rows: List[Dict[str, Any]] = []
    for _, r in df.iterrows():
        raw_date = str(r.get(date_col, "")).strip() if date_col else ""
        parsed_dt = pd.to_datetime(raw_date, errors="coerce")
        date_str = parsed_dt.strftime("%Y-%m-%d") if pd.notna(parsed_dt) else (raw_date or "Unset")

        desc = str(r.get(desc_col, "")).strip()
        if not desc or desc.lower() in {"nan", "none"}:
            continue

        signed_amount: Optional[float] = None
        if amount_col:
            amt = parse_amount(r.get(amount_col))
            if amt is not None:
                signed_amount = amt
        elif debit_col or credit_col:
            dr = parse_amount(r.get(debit_col)) if debit_col else None
            cr = parse_amount(r.get(credit_col)) if credit_col else None
            if cr is not None and cr > 0:
                signed_amount = abs(cr)
            elif dr is not None and dr != 0:
                signed_amount = -abs(dr)
        elif qty_col and price_col:
            q = parse_amount(r.get(qty_col))
            p = parse_amount(r.get(price_col))
            if q is not None and p is not None:
                signed_amount = q * p

        if signed_amount is None or signed_amount == 0:
            continue

        direction = "inflow" if signed_amount > 0 else "outflow"
        abs_amount = round(abs(signed_amount), 2)
        category = categorize_description(desc)

        rows.append({
            "date": date_str,
            "description": desc,
            "amount": abs_amount,
            "signed_amount": round(signed_amount, 2),
            "direction": direction,
            "category": category,
        })

    return rows


def parse_pdf_bytes(content: bytes) -> List[Dict[str, Any]]:
    extracted_rows: List[Dict[str, Any]] = []
    date_pattern = re.compile(
        r"^(\d{4}-\d{2}-\d{2}|\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\w{3}\s+\d{1,2},?\s*\d{0,4})\s+(.+?)\s+(-?\(?\$?[\d,]+\.\d{2}\)?)$"
    )

    with pdfplumber.open(io.BytesIO(content)) as pdf:
        for page in pdf.pages:
            tables = page.extract_tables() or []
            for table in tables:
                if len(table) > 1:
                    headers = [h if h else f"col_{i}" for i, h in enumerate(table[0])]
                    t_df = pd.DataFrame(table[1:], columns=headers)
                    extracted_rows.extend(normalize_dataframe(t_df))

            if not extracted_rows:
                text = page.extract_text() or ""
                for line in text.splitlines():
                    m = date_pattern.match(line.strip())
                    if m:
                        dt_raw, desc, amt_raw = m.groups()
                        amt = parse_amount(amt_raw)
                        if amt is not None and amt != 0:
                            parsed_dt = pd.to_datetime(dt_raw, errors="coerce")
                            dt_str = parsed_dt.strftime("%Y-%m-%d") if pd.notna(parsed_dt) else dt_raw
                            extracted_rows.append({
                                "date": dt_str,
                                "description": desc.strip(),
                                "amount": round(abs(amt), 2),
                                "signed_amount": round(amt, 2),
                                "direction": "inflow" if amt > 0 else "outflow",
                                "category": categorize_description(desc),
                            })

    return extracted_rows


def parse_uploaded_file(filename: str, content: bytes) -> List[Dict[str, Any]]:
    ext = os.path.splitext(filename.lower())[1]
    if ext == ".csv":
        df = pd.read_csv(io.BytesIO(content))
        return normalize_dataframe(df)
    if ext in {".xlsx", ".xls"}:
        df = pd.read_excel(io.BytesIO(content))
        return normalize_dataframe(df)
    if ext == ".pdf":
        return parse_pdf_bytes(content)
    raise HTTPException(status_code=400, detail="Unsupported file type. Upload a .csv, .xlsx, .xls, or .pdf file.")


def analyze_transactions(
    transactions: List[Dict[str, Any]],
    currency: str = "$",
    check_duplicates: bool = True,
) -> Dict[str, Any]:
    cur = (currency or "$").strip() or "$"
    if not transactions:
        return {
            "currency": cur,
            "summary": {
                "total_inflow": 0.0,
                "total_outflow": 0.0,
                "net_cash_flow": 0.0,
                "flagged_items_count": 0,
                "flagged_total": 0.0,
                "transaction_count": 0,
            },
            "categories": [],
            "flagged_items": [],
            "plain_english_brief": ["No valid records were found in the uploaded file."],
            "transactions": [],
        }

    df = pd.DataFrame(transactions)
    inflow = float(df[df["direction"] == "inflow"]["amount"].sum()) if "inflow" in df["direction"].values else 0.0
    outflow = float(df[df["direction"] == "outflow"]["amount"].sum()) if "outflow" in df["direction"].values else 0.0
    net_cash = round(inflow - outflow, 2)

    flagged: List[Dict[str, Any]] = []
    for _, row in df.iterrows():
        desc_l = str(row["description"]).lower()
        if row["category"] == "Fees & Bank Charges" or any(k in desc_l for k in FEE_KEYWORDS):
            flagged.append({
                "date": row["date"],
                "description": row["description"],
                "amount": row["amount"],
                "reason": "Bank/service fee",
            })

    if check_duplicates:
        out_df = df[df["direction"] == "outflow"]
        if not out_df.empty:
            dupes = out_df[out_df.duplicated(subset=["date", "amount"], keep=False)]
            for _, drow in dupes.iterrows():
                flagged.append({
                    "date": drow["date"],
                    "description": drow["description"],
                    "amount": drow["amount"],
                    "reason": "Potential duplicate debit",
                })

    cat_summary = (
        df[df["direction"] == "outflow"]
        .groupby("category", as_index=False)["amount"]
        .sum()
        .sort_values("amount", ascending=False)
    )
    categories = [
        {"category": r["category"], "total_outflow": round(float(r["amount"]), 2)}
        for _, r in cat_summary.iterrows()
    ]

    flagged_total = round(sum(item["amount"] for item in flagged), 2) if flagged else 0.0
    top_cat_text = (
        f"Top outflow category is {categories[0]['category']} at {cur}{categories[0]['total_outflow']:,.2f}."
        if categories
        else "No outflow categories recorded."
    )
    cash_status = "positive" if net_cash >= 0 else "negative"

    brief = [
        f"Net cash flow is {cash_status} at {cur}{net_cash:,.2f} ({cur}{inflow:,.2f} inflow vs. {cur}{outflow:,.2f} outflow).",
        top_cat_text,
        f"Flagged {len(flagged)} fee/duplicate item(s) totaling {cur}{flagged_total:,.2f} for review.",
    ]

    return {
        "currency": cur,
        "summary": {
            "total_inflow": round(inflow, 2),
            "total_outflow": round(outflow, 2),
            "net_cash_flow": net_cash,
            "flagged_items_count": len(flagged),
            "flagged_total": flagged_total,
            "transaction_count": len(transactions),
        },
        "categories": categories,
        "flagged_items": flagged,
        "plain_english_brief": brief,
        "transactions": transactions,
    }


def record_history(source_name: str, report: Dict[str, Any]) -> Dict[str, Any]:
    entry = {
        "id": f"run-{int(datetime.now(timezone.utc).timestamp() * 1000)}",
        "source_name": source_name,
        "analyzed_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "report": report,
    }
    RECENT_HISTORY.insert(0, entry)
    if len(RECENT_HISTORY) > MAX_HISTORY:
        del RECENT_HISTORY[MAX_HISTORY:]
    return entry


SAMPLE_TRANSACTIONS = [
    {"date": "2026-11-01", "description": "Client Retainer - Acme Retail", "amount": 3200.00, "signed_amount": 3200.00, "direction": "inflow", "category": "Sales & Revenue"},
    {"date": "2026-11-02", "description": "Monthly Bank Maintenance Fee", "amount": 28.50, "signed_amount": -28.50, "direction": "outflow", "category": "Fees & Bank Charges"},
    {"date": "2026-11-03", "description": "Cloud Hosting & Database Plan", "amount": 64.00, "signed_amount": -64.00, "direction": "outflow", "category": "Software & Subscriptions"},
    {"date": "2026-11-05", "description": "Paystack Settlement Batch #409", "amount": 1850.00, "signed_amount": 1850.00, "direction": "inflow", "category": "Sales & Revenue"},
    {"date": "2026-11-06", "description": "Office Supplies Vendor", "amount": 145.00, "signed_amount": -145.00, "direction": "outflow", "category": "Supplies & Inventory"},
    {"date": "2026-11-06", "description": "Office Supplies Vendor", "amount": 145.00, "signed_amount": -145.00, "direction": "outflow", "category": "Supplies & Inventory"},
    {"date": "2026-11-10", "description": "SMS Alert & Transfer Fee", "amount": 14.00, "signed_amount": -14.00, "direction": "outflow", "category": "Fees & Bank Charges"},
    {"date": "2026-11-15", "description": "Workspace Rent Allocation", "amount": 650.00, "signed_amount": -650.00, "direction": "outflow", "category": "Rent & Operations"},
]


@app.get("/health")
async def health() -> JSONResponse:
    return JSONResponse({"status": "ok", "service": "kevsight"})


@app.get("/api/history")
async def get_history() -> JSONResponse:
    return JSONResponse({"items": RECENT_HISTORY})


@app.delete("/api/history")
async def clear_history() -> JSONResponse:
    RECENT_HISTORY.clear()
    return JSONResponse({"ok": True})


@app.get("/", response_class=HTMLResponse)
async def home() -> str:
    return """
<!DOCTYPE html>
<html lang="en" data-theme="light">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Kevsight — Finance & Sales Data Intelligence</title>
  <style>
    :root {
      --bg: #f6f5fa;
      --surface: #ffffff;
      --surface-alt: #f1eff8;
      --text: #14131a;
      --muted: #5c5870;
      --border: #e3dfef;
      --accent: #5b21b6;
      --accent-hover: #4c1d95;
      --accent-soft: rgba(91, 33, 182, 0.1);
      --inflow: #15803d;
      --inflow-bg: rgba(21, 128, 61, 0.12);
      --outflow: #b91c1c;
      --outflow-bg: rgba(185, 28, 28, 0.12);
      --warn: #b45309;
      --warn-bg: rgba(180, 83, 9, 0.12);
      --shadow: 0 6px 24px rgba(20, 19, 26, 0.05);
    }

    [data-theme="dark"] {
      --bg: #0e0d14;
      --surface: #171522;
      --surface-alt: #211e30;
      --text: #f3f1fa;
      --muted: #a49eb8;
      --border: #2c2840;
      --accent: #8b5cf6;
      --accent-hover: #7c3aed;
      --accent-soft: rgba(139, 92, 246, 0.16);
      --inflow: #4ade80;
      --inflow-bg: rgba(74, 222, 128, 0.14);
      --outflow: #f87171;
      --outflow-bg: rgba(248, 113, 113, 0.14);
      --warn: #fbbf24;
      --warn-bg: rgba(251, 191, 36, 0.14);
      --shadow: 0 10px 30px rgba(0, 0, 0, 0.35);
    }

    * { box-sizing: border-box; }
    body {
      font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      margin: 0;
      background: var(--bg);
      color: var(--text);
      transition: background-color 0.2s ease, color 0.2s ease;
      line-height: 1.45;
    }

    .container { max-width: 1120px; margin: 24px auto 64px; padding: 0 18px; }

    header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 12px;
      flex-wrap: wrap;
      margin-bottom: 16px;
    }

    .brand-wrap { display: flex; align-items: center; gap: 12px; }
    .brand-mark {
      width: 40px; height: 40px; border-radius: 12px;
      background: linear-gradient(135deg, var(--accent), #a855f7);
      display: grid; place-items: center; color: #fff; font-weight: 800; font-size: 19px;
    }
    .brand-title { font-size: 22px; font-weight: 800; letter-spacing: -0.02em; }
    .brand-sub { font-size: 12px; color: var(--muted); }

    .header-actions { display: flex; gap: 8px; align-items: center; flex-wrap: wrap; }

    .nav-bar {
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
      padding: 6px;
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 14px;
      margin-bottom: 18px;
      box-shadow: var(--shadow);
    }
    .nav-btn {
      background: transparent;
      color: var(--muted);
      border: none;
      padding: 9px 14px;
      border-radius: 10px;
      font-weight: 600;
      font-size: 14px;
      cursor: pointer;
    }
    .nav-btn:hover { background: var(--surface-alt); color: var(--text); }
    .nav-btn.active { background: var(--accent); color: #fff; }

    button {
      background: var(--accent);
      color: #fff;
      border: none;
      padding: 10px 15px;
      border-radius: 10px;
      cursor: pointer;
      font-weight: 600;
      font-size: 14px;
      transition: background-color 0.15s ease, transform 0.1s ease;
    }
    button:hover { background: var(--accent-hover); }
    button:active { transform: scale(0.99); }
    button:disabled { opacity: 0.6; cursor: not-allowed; }

    button.ghost {
      background: var(--surface);
      color: var(--text);
      border: 1px solid var(--border);
    }
    button.ghost:hover { background: var(--surface-alt); }

    .card {
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 16px;
      padding: 20px;
      margin-bottom: 18px;
      box-shadow: var(--shadow);
    }

    .upload-box {
      display: flex;
      gap: 12px;
      flex-wrap: wrap;
      align-items: center;
      justify-content: space-between;
    }
    .upload-left { display: flex; gap: 10px; flex-wrap: wrap; align-items: center; }
    .file-input {
      background: var(--surface-alt);
      color: var(--text);
      border: 1px dashed var(--border);
      padding: 9px 12px;
      border-radius: 10px;
      font-size: 13px;
      max-width: 100%;
    }
    .hint { font-size: 12px; color: var(--muted); margin-top: 8px; }

    .grid-4 {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
      gap: 14px;
      margin-bottom: 18px;
    }
    .kpi-card {
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 14px;
      padding: 16px 18px;
      box-shadow: var(--shadow);
    }
    .kpi-label { font-size: 11px; font-weight: 700; color: var(--muted); text-transform: uppercase; letter-spacing: 0.05em; }
    .kpi-val { font-size: 25px; font-weight: 800; margin-top: 6px; letter-spacing: -0.02em; }
    .kpi-meta { font-size: 12px; color: var(--muted); margin-top: 4px; }

    .grid-2 {
      display: grid;
      grid-template-columns: 1.35fr 1fr;
      gap: 18px;
      margin-bottom: 18px;
    }
    @media (max-width: 820px) {
      .grid-2 { grid-template-columns: 1fr; }
    }

    .section-head {
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 10px;
      flex-wrap: wrap;
      margin-bottom: 14px;
    }
    .section-title { margin: 0; font-size: 16px; font-weight: 700; }

    .flow-bars { display: flex; flex-direction: column; gap: 12px; margin-bottom: 18px; }
    .bar-row { display: flex; flex-direction: column; gap: 5px; }
    .bar-top { display: flex; justify-content: space-between; font-size: 13px; font-weight: 600; }
    .bar-track { height: 10px; background: var(--surface-alt); border-radius: 999px; overflow: hidden; }
    .bar-fill { height: 100%; border-radius: 999px; transition: width 0.35s ease; }

    .cat-list { display: flex; flex-direction: column; gap: 10px; }
    .cat-item { display: flex; flex-direction: column; gap: 4px; }
    .cat-meta { display: flex; justify-content: space-between; font-size: 13px; }
    .cat-name { font-weight: 600; }
    .cat-amt { color: var(--muted); font-variant-numeric: tabular-nums; }

    .brief-list { margin: 0; padding-left: 18px; display: flex; flex-direction: column; gap: 8px; }
    .brief-list li { font-size: 14px; }

    .flag-box {
      margin-top: 14px;
      padding: 12px 14px;
  
