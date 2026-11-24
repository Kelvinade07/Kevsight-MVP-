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
    currency: str = "₦"
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
        (c for c in df.columns if any(k in c for k in ["date", "time", "posted", "order_date", "transaction_day"])),
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
            page_rows: List[Dict[str, Any]] = []
            tables = page.extract_tables() or []
            for table in tables:
                if len(table) > 1:
                    headers = [h if h else f"col_{i}" for i, h in enumerate(table[0])]
                    t_df = pd.DataFrame(table[1:], columns=headers)
                    page_rows.extend(normalize_dataframe(t_df))

            # Some statements have tables on only selected pages. Fall back to text
            # extraction for each page that did not yield usable table rows.
            if not page_rows:
                text = page.extract_text() or ""
                for line in text.splitlines():
                    m = date_pattern.match(line.strip())
                    if m:
                        dt_raw, desc, amt_raw = m.groups()
                        amt = parse_amount(amt_raw)
                        if amt is not None and amt != 0:
                            parsed_dt = pd.to_datetime(dt_raw, errors="coerce")
                            dt_str = parsed_dt.strftime("%Y-%m-%d") if pd.notna(parsed_dt) else dt_raw
                            page_rows.append({
                                "date": dt_str,
                                "description": desc.strip(),
                                "amount": round(abs(amt), 2),
                                "signed_amount": round(amt, 2),
                                "direction": "inflow" if amt > 0 else "outflow",
                                "category": categorize_description(desc),
                            })
            extracted_rows.extend(page_rows)

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
    currency: str = "₦",
    check_duplicates: bool = True,
) -> Dict[str, Any]:
    cur = (currency or "₦").strip() or "₦"
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

    # Build flags per transaction row so a transaction that is both a fee
    # and a potential duplicate is counted only once in the flagged total.
    flag_reasons: Dict[int, List[str]] = {}
    for idx, row in df.iterrows():
        desc_l = str(row["description"]).lower()
        if row["category"] == "Fees & Bank Charges" or any(k in desc_l for k in FEE_KEYWORDS):
            flag_reasons.setdefault(int(idx), []).append("Bank/service fee")

    if check_duplicates:
        out_df = df[df["direction"] == "outflow"].copy()
        if not out_df.empty:
            out_df["_description_key"] = (
                out_df["description"].astype(str).str.lower()
                .str.replace(r"\\s+", " ", regex=True).str.strip()
            )
            duplicate_rows = out_df[
                out_df.duplicated(
                    subset=["date", "amount", "_description_key"],
                    keep=False
                )
            ]
            for idx in duplicate_rows.index:
                flag_reasons.setdefault(int(idx), []).append("Potential duplicate debit")

    flagged: List[Dict[str, Any]] = []
    for idx, reasons in flag_reasons.items():
        row = df.loc[idx]
        flagged.append({
            "date": row["date"],
            "description": row["description"],
            "amount": row["amount"],
            "reason": " + ".join(reasons),
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
  border-radius: 12px;
      background: var(--warn-bg);
      border: 1px solid rgba(180, 83, 9, 0.25);
      font-size: 13px;
    }
    .flag-title { font-weight: 700; color: var(--warn); margin-bottom: 6px; }

    .controls-row { display: flex; gap: 8px; flex-wrap: wrap; align-items: center; }
    .search-input, .select-input, .text-input {
      background: var(--surface-alt);
      color: var(--text);
      border: 1px solid var(--border);
      padding: 8px 12px;
      border-radius: 9px;
      font-size: 13px;
    }
    .search-input { min-width: 200px; }

    .table-wrap { overflow-x: auto; }
    table { width: 100%; border-collapse: collapse; font-size: 14px; }
    th, td { text-align: left; padding: 11px 10px; border-bottom: 1px solid var(--border); white-space: nowrap; }
    td.desc-col { white-space: normal; min-width: 180px; }
    th { color: var(--muted); font-size: 11px; text-transform: uppercase; letter-spacing: 0.05em; }

    .badge {
      display: inline-block;
      padding: 3px 9px;
      border-radius: 999px;
      font-size: 12px;
      font-weight: 600;
    }
    .badge-inflow { color: var(--inflow); background: var(--inflow-bg); }
    .badge-outflow { color: var(--outflow); background: var(--outflow-bg); }
    .badge-cat { color: var(--accent); background: var(--accent-soft); }
    .amt-in { color: var(--inflow); font-weight: 700; font-variant-numeric: tabular-nums; }
    .amt-out { color: var(--outflow); font-weight: 700; font-variant-numeric: tabular-nums; }

    .settings-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
      gap: 16px;
    }
    .field { display: flex; flex-direction: column; gap: 6px; }
    .field label { font-size: 12px; font-weight: 700; color: var(--muted); text-transform: uppercase; letter-spacing: 0.04em; }

    .history-item {
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 12px;
      flex-wrap: wrap;
      padding: 14px;
      border: 1px solid var(--border);
      border-radius: 12px;
      background: var(--surface-alt);
      margin-bottom: 10px;
    }

    .faq-item {
      padding: 12px 0;
      border-bottom: 1px solid var(--border);
    }
    .faq-item:last-child { border-bottom: none; }
    .faq-q { font-weight: 700; margin-bottom: 4px; }

    .toast {
      position: fixed;
      bottom: 20px;
      right: 20px;
      background: var(--surface);
      color: var(--text);
      border: 1px solid var(--border);
      box-shadow: var(--shadow);
      padding: 11px 16px;
      border-radius: 12px;
      font-size: 13px;
      font-weight: 600;
      z-index: 50;
      display: none;
    }

    /* Kevsight UI refresh */
    :root {
      --bg: #f5f7fb;
      --surface: #ffffff;
      --surface-alt: #f3f5fa;
      --text: #172033;
      --muted: #68738a;
      --border: #e5eaf2;
      --accent: #3157d5;
      --accent-hover: #2446b8;
      --accent-soft: rgba(49, 87, 213, .09);
      --shadow: 0 10px 30px rgba(22, 34, 58, .055);
    }
    [data-theme="dark"] {
      --bg: #0c1220;
      --surface: #131d30;
      --surface-alt: #1a2740;
      --text: #edf2ff;
      --muted: #a2aec4;
      --border: #293650;
      --accent: #7895ff;
      --accent-hover: #91a7ff;
      --accent-soft: rgba(120, 149, 255, .13);
      --shadow: 0 12px 34px rgba(0, 0, 0, .2);
    }
    body { background: var(--bg); letter-spacing: -.01em; }
    .container { max-width: 1320px; margin: 28px auto 70px; padding: 0 28px; }
    header { padding: 4px 0 20px; border-bottom: 1px solid var(--border); margin-bottom: 20px; }
    .brand-mark { border-radius: 13px; background: linear-gradient(145deg, #2347b9, #6485ff); box-shadow: 0 8px 18px rgba(49,87,213,.22); }
    .brand-title { letter-spacing: -.045em; }
    .brand-sub { margin-top: 2px; }
    .nav-bar { border-radius: 13px; padding: 5px; box-shadow: none; }
    .nav-btn { padding: 10px 16px; }
    .nav-btn.active { box-shadow: 0 4px 10px rgba(49,87,213,.2); }
    button { border-radius: 10px; }
    .card, .kpi-card { border-radius: 17px; box-shadow: var(--shadow); }
    .card { padding: 22px; }
    .kpi-card { padding: 19px 20px; }
    .kpi-label { letter-spacing: .075em; }
    .kpi-val { letter-spacing: -.045em; font-size: clamp(21px, 2vw, 28px); }
    .page-intro { display:flex; align-items:flex-end; justify-content:space-between; gap:18px; margin: 25px 0 20px; }
    .page-intro h1 { margin:7px 0 6px; font-size: clamp(25px, 3vw, 35px); line-height:1.12; letter-spacing:-.055em; }
    .page-intro p { margin:0; max-width:690px; color:var(--muted); font-size:14px; }
    .eyebrow { color:var(--accent); font-size:10px; font-weight:800; letter-spacing:.14em; }
    .intro-note { white-space:nowrap; color:var(--muted); font-size:12px; border:1px solid var(--border); border-radius:999px; padding:8px 12px; background:var(--surface); }
    .status-dot { display:inline-block; width:7px; height:7px; border-radius:50%; background:#22a06b; margin-right:7px; }
    .upload-box { align-items:center; }
    .upload-left { width:100%; flex:1; }
    .dropzone { min-height:84px; flex:1; display:flex; align-items:center; gap:13px; padding:14px 16px; border:1.5px dashed #aebbd2; border-radius:13px; background:var(--surface-alt); cursor:pointer; transition:.18s ease; }
    .dropzone:hover, .dropzone.dragover { border-color:var(--accent); background:var(--accent-soft); }
    .dropzone input { position:absolute; width:1px; height:1px; opacity:0; pointer-events:none; }
    .drop-icon { width:38px; height:38px; display:grid; place-items:center; flex:0 0 38px; border-radius:11px; background:var(--accent-soft); color:var(--accent); font-size:25px; font-weight:500; }
    #fileLabel { font-size:13px; }
    .upload-left > button { min-height:45px; padding:0 22px; }
    .hint { line-height:1.5; }
    .section-title { letter-spacing:-.025em; }
    th { font-weight:750; }
    tbody tr:hover { background:var(--surface-alt); }
    .search-input, .select-input, .text-input { border-radius:10px; min-height:38px; }
    button:focus-visible, input:focus-visible, select:focus-visible, .dropzone:focus-visible { outline:3px solid var(--accent-soft); outline-offset:2px; }
    .toast { border-radius:12px; }
    @media (max-width:700px) {
      .container { margin:16px auto 40px; padding:0 13px; }
      header { align-items:flex-start; }
      .page-intro { align-items:flex-start; flex-direction:column; margin:20px 0 16px; }
      .intro-note { display:none; }
      .card { padding:16px; }
      .upload-left { flex-direction:column; align-items:stretch; }
      .dropzone { width:100%; }
      .upload-left > button { width:100%; }
      .nav-bar { overflow-x:auto; flex-wrap:nowrap; }
      .nav-btn { white-space:nowrap; }
      .grid-4 { grid-template-columns:repeat(2, minmax(0,1fr)); gap:9px; }
      .kpi-card { padding:13px; }
      .kpi-val { font-size:19px; overflow-wrap:anywhere; }
      .kpi-label { font-size:9px; }
      .controls-row { width:100%; }
      .search-input { min-width:0; flex:1; }
      .section-head { align-items:flex-start; }
    }
    @media (max-width:390px) {
      .grid-4 { grid-template-columns:1fr; }
    }
  </style>
</head>
<body>
  <div class="container">
    <header>
      <div class="brand-wrap">
        <div class="brand-mark">K</div>
        <div>
          <div class="brand-title" id="brandDisplay">Kevsight</div>
          <div class="brand-sub">Deterministic Finance &amp; Sales Data Intelligence</div>
        </div>
      </div>
      <div class="header-actions">
        <button type="button" class="ghost" id="themeBtn" aria-label="Toggle theme">Dark Mode</button>
        <button type="button" id="sampleBtn">Try Sample Data</button>
      </div>
    </header>

    <nav class="nav-bar" aria-label="Primary navigation">
      <button type="button" class="nav-btn active" data-view="analyzeView">Analyze</button>
      <button type="button" class="nav-btn" data-view="historyView">Data History</button>
      <button type="button" class="nav-btn" data-view="settingsView">Settings</button>
      <button type="button" class="nav-btn" data-view="supportView">Support</button>
    </nav>

    <!-- ANALYZE VIEW -->
    <div id="analyzeView">
      <section class="page-intro">
        <div>
          <div class="eyebrow">FINANCIAL DATA WORKSPACE</div>
          <h1>Turn raw transactions into clear decisions.</h1>
          <p>Upload a statement or sales export to review cash flow, spending categories and items that may need attention.</p>
        </div>
        <div class="intro-note"><span class="status-dot"></span> Ready for analysis</div>
      </section>
      <section class="card">
        <form id="uploadForm" class="upload-box">
          <div class="upload-left">
            <div class="dropzone" id="dropzone" tabindex="0" role="button" aria-label="Choose or drop a data file">
              <div class="drop-icon">↑</div>
              <div>
                <strong id="fileLabel">Choose a file or drag it here</strong>
                <div class="hint" style="margin:3px 0 0;">CSV, XLSX, XLS or text-based PDF · up to 15 MB</div>
              </div>
              <input type="file" id="fileInput" accept=".csv,.xlsx,.xls,.pdf" required />
            </div>
            <button type="submit" id="analyzeBtn">Analyze Data</button>
          </div>
          <div class="controls-row">
            <button type="button" class="ghost" id="exportCsvBtn">Export CSV</button>
          </div>
        </form>
        <div class="hint">Supports financial statements, sales exports, and transaction logs in CSV, Excel (.xlsx/.xls), and text-based PDF.</div>
      </section>

      <section class="grid-4">
        <div class="kpi-card">
          <div class="kpi-label">Total Inflow / Sales</div>
          <div class="kpi-val amt-in" id="inflow">$0.00</div>
          <div class="kpi-meta" id="txCountMeta">0 records analyzed</div>
        </div>
        <div class="kpi-card">
          <div class="kpi-label">Total Outflow</div>
          <div class="kpi-val amt-out" id="outflow">$0.00</div>
          <div class="kpi-meta">Expenses, fees &amp; payouts</div>
        </div>
        <div class="kpi-card">
          <div class="kpi-label">Net Cash Flow</div>
          <div class="kpi-val" id="net">$0.00</div>
          <div class="kpi-meta" id="netMeta">Awaiting data</div>
        </div>
        <div class="kpi-card">
          <div class="kpi-label">Flagged Items</div>
          <div class="kpi-val" id="flagged">0</div>
          <div class="kpi-meta" id="flaggedMeta">$0.00 under review</div>
        </div>
      </section>

      <section class="grid-2">
        <div class="card" style="margin-bottom:0;">
          <div class="section-head">
            <h2 class="section-title">Cash Flow &amp; Outflow Breakdown</h2>
          </div>

          <div class="flow-bars">
            <div class="bar-row">
              <div class="bar-top">
                <span>Inflow / Sales</span>
                <span id="inflowBarLabel">$0.00</span>
              </div>
              <div class="bar-track">
                <div class="bar-fill" id="inflowBar" style="width: 0%; background: var(--inflow);"></div>
              </div>
            </div>
            <div class="bar-row">
              <div class="bar-top">
                <span>Outflow</span>
                <span id="outflowBarLabel">$0.00</span>
              </div>
              <div class="bar-track">
                <div class="bar-fill" id="outflowBar" style="width: 0%; background: var(--outflow);"></div>
              </div>
            </div>
          </div>

          <div class="kpi-label" style="margin-bottom:8px;">Top Outflow Categories</div>
          <div class="cat-list" id="catList">
            <div class="hint" style="margin:0;">Load sample data or analyze a file to view category visuals.</div>
          </div>
        </div>

        <div class="card" style="margin-bottom:0;">
          <div class="section-head">
            <h2 class="section-title">Plain-English Brief</h2>
          </div>
          <ul class="brief-list" id="briefList">
            <li>Click <strong>Try Sample Data</strong> or upload a file and click <strong>Analyze Data</strong>.</li>
          </ul>
          <div class="flag-box" id="flagBox" style="display:none;">
            <div class="flag-title">Flagged Fees &amp; Potential Duplicates</div>
            <div id="flagItems"></div>
          </div>
        </div>
      </section>

      <section class="card" style="margin-top:18px;">
        <div class="section-head">
          <h2 class="section-title">Normalized Records</h2>
          <div class="controls-row">
            <select id="typeFilter" class="select-input" aria-label="Filter by type">
              <option value="all">All Types</option>
              <option value="inflow">Inflow / Sales</option>
              <option value="outflow">Outflow</option>
            </select>
            <input type="text" id="searchInput" class="search-input" placeholder="Filter description or category..." />
          </div>
        </div>
        <div class="table-wrap">
          <table>
            <thead>
              <tr>
                <th>Date</th>
                <th>Description</th>
                <th>Category</th>
                <th>Type</th>
                <th>Amount</th>
              </tr>
            </thead>
            <tbody id="txBody"></tbody>
          </table>
        </div>
      </section>
    </div>

    <!-- HISTORY VIEW -->
    <div id="historyView" style="display:none;">
      <section class="card">
        <div class="section-head">
          <div>
            <h2 class="section-title">Data History</h2>
            <div class="hint" style="margin-top:4px;">Reload any recent sample or uploaded analysis snapshot in one click.</div>
          </div>
          <button type="button" class="ghost" id="clearHistoryBtn">Clear History</button>
        </div>
        <div id="historyList">
          <div class="hint">No analyzed datasets yet. Run sample data or upload a file to build history.</div>
        </div>
      </section>
    </div>

    <!-- SETTINGS VIEW -->
    <div id="settingsView" style="display:none;">
      <section class="card">
        <div class="section-head">
          <h2 class="section-title">Workspace Settings</h2>
        </div>
        <div class="settings-grid">
          <div class="field">
            <label for="setWorkspace">Workspace Label</label>
            <input type="text" id="setWorkspace" class="text-input" placeholder="Kevsight" />
          </div>
          <div class="field">
            <label for="setCurrency">Display Currency Symbol</label>
            <input type="text" id="setCurrency" class="text-input" maxlength="8" placeholder="₦" />
          </div>
          <div class="field">
            <label for="setDuplicates">Flag Same-Day Duplicate Debits</label>
            <select id="setDuplicates" class="select-input">
              <option value="true">Enabled (Recommended)</option>
              <option value="false">Disabled</option>
            </select>
          </div>
          <div class="field">
            <label for="setTheme">Appearance Theme</label>
            <select id="setTheme" class="select-input">
              <option value="light">Light Mode</option>
              <option value="dark">Dark Mode</option>
            </select>
          </div>
        </div>
        <div style="margin-top:16px; display:flex; gap:10px; flex-wrap:wrap;">
          <button type="button" id="saveSettingsBtn">Save Settings</button>
          <button type="button" class="ghost" id="resetSettingsBtn">Reset Defaults</button>
        </div>
      </section>
    </div>

    <!-- SUPPORT VIEW -->
    <div id="supportView" style="display:none;">
      <section class="card">
        <div class="section-head">
          <h2 class="section-title">Support &amp; Data Guide</h2>
          <button type="button" class="ghost" id="healthCheckBtn">Check Service Health</button>
        </div>
        <div id="healthResult" class="hint" style="margin-bottom:12px;"></div>

        <div class="faq-item">
          <div class="faq-q">What file formats and columns work best?</div>
          <div class="hint" style="margin:0;">Upload <strong>.csv</strong>, <strong>.xlsx</strong>, <strong>.xls</strong>, or text-based <strong>.pdf</strong> files containing a date column, a description/item/customer column, and an amount (or separate debit/credit or quantity + unit price columns).</div>
        </div>
        <div class="faq-item">
          <div class="faq-q">Can I analyze sales data as well as bank/expense statements?</div>
          <div class="hint" style="margin:0;">Yes. Positive amounts or credit rows are treated as inflows/sales, while outflows are grouped into categories, checked for fees/duplicates, and summarized in plain English.</div>
        </div>
        <div class="faq-item">
          <div class="faq-q">Why does the free deployment sometimes take a moment on first load?</div>
          <div class="hint" style="margin:0;">Free-tier hosting instances may sleep after idle periods and wake automatically when a request arrives.</div>
        </div>
      </section>
    </div>
  </div>

  <div class="toast" id="toast"></div>

  <script>
    const SETTINGS_KEY = "kevsight_settings_v1";
    const LOCAL_HISTORY_KEY = "kevsight_history_v1";

    const DEFAULT_SETTINGS = {
      workspace: "Kevsight",
      currency: "₦",
      checkDuplicates: true,
      theme: "light"
    };

    let settings = loadSettings();
    let localHistory = loadLocalHistory();
    let currentTransactions = [];
    let lastReportData = null;

    function showToast(msg) {
      const el = document.getElementById("toast");
      el.textContent = msg;
      el.style.display = "block";
      clearTimeout(showToast._t);
      showToast._t = setTimeout(() => { el.style.display = "none"; }, 2600);
    }

    function loadSettings() {
      try {
        return { ...DEFAULT_SETTINGS, ...JSON.parse(localStorage.getItem(SETTINGS_KEY) || "{}") };
      } catch (_) {
        return { ...DEFAULT_SETTINGS };
      }
    }

    function saveSettingsToStorage() {
      localStorage.setItem(SETTINGS_KEY, JSON.stringify(settings));
    }

    function loadLocalHistory() {
      try {
        const parsed = JSON.parse(localStorage.getItem(LOCAL_HISTORY_KEY) || "[]");
        return Array.isArray(parsed) ? parsed : [];
      } catch (_) {
        return [];
      }
    }

    function saveLocalHistory() {
      localStorage.setItem(LOCAL_HISTORY_KEY, JSON.stringify(localHistory.slice(0, 20)));
    }

    // Escape all uploaded/user-provided text before inserting it into HTML.
    function escapeHtml(value) {
      return String(value ?? "").replace(/[&<>"']/g, (ch) => ({
        "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"
      }[ch]));
    }

    const fmt = (n, cur = settings.currency) => (cur || "₦") + Number(n || 0).toLocaleString(undefined, {
      minimumFractionDigits: 2,
      maximumFractionDigits: 2
    });

    function applyTheme(theme) {
      const next = theme === "dark" ? "dark" : "light";
      document.documentElement.setAttribute("data-theme", next);
      document.getElementById("themeBtn").textContent = next === "dark" ? "Light Mode" : "Dark Mode";
      document.getElementById("setTheme").value = next;
      settings.theme = next;
      saveSettingsToStorage();
    }

    function syncSettingsInputs() {
      document.getElementById("brandDisplay").textContent = settings.workspace || "Kevsight";
      document.getElementById("setWorkspace").value = settings.workspace || "Kevsight";
      document.getElementById("setCurrency").value = settings.currency || "₦";
      document.getElementById("setDuplicates").value = String(Boolean(settings.checkDuplicates));
      document.getElementById("setTheme").value = settings.theme || "light";
    }

    function switchView(viewId) {
      ["analyzeView", "historyView", "settingsView", "supportView"].forEach((id) => {
        document.getElementById(id).style.display = id === viewId ? "block" : "none";
      });
      document.querySelectorAll(".nav-btn").forEach((btn) => {
        btn.classList.toggle("active", btn.dataset.view === viewId);
      });
      if (viewId === "historyView") {
        renderHistory();
      }
    }

    document.querySelectorAll(".nav-btn").forEach((btn) => {
      btn.addEventListener("click", () => switchView(btn.dataset.view));
    });

    document.getElementById("themeBtn").addEventListener("click", () => {
      applyTheme(settings.theme === "dark" ? "light" : "dark");
    });

    document.getElementById("saveSettingsBtn").addEventListener("click", () => {
      settings.workspace = (document.getElementById("setWorkspace").value || "Kevsight").trim();
      settings.currency = (document.getElementById("setCurrency").value || "₦").trim();
      settings.checkDuplicates = document.getElementById("setDuplicates").value === "true";
      applyTheme(document.getElementById("setTheme").value);
      saveSettingsToStorage();
      syncSettingsInputs();
      if (lastReportData) renderReport(lastReportData);
      showToast("Settings saved");
    });

    document.getElementById("resetSettingsBtn").addEventListener("click", () => {
      settings = { ...DEFAULT_SETTINGS };
      saveSettingsToStorage();
      applyTheme(settings.theme);
      syncSettingsInputs();
      if (lastReportData) renderReport(lastReportData);
      showToast("Settings reset to defaults");
    });

    function renderTable(rows) {
      const txBody = document.getElementById("txBody");
      txBody.innerHTML = "";
      if (!rows.length) {
        txBody.innerHTML = `<tr><td colspan="5" style="color:var(--muted);">No matching records to display.</td></tr>`;
        return;
      }
      rows.forEach((tx) => {
        const isIn = tx.direction === "inflow";
        const tr = document.createElement("tr");
        tr.innerHTML = `
          <td>${escapeHtml(tx.date)}</td>
          <td class="desc-col">${escapeHtml(tx.description)}</td>
          <td><span class="badge badge-cat">${escapeHtml(tx.category)}</span></td>
          <td><span class="badge ${isIn ? "badge-inflow" : "badge-outflow"}">${escapeHtml(tx.direction)}</span></td>
          <td class="${isIn ? "amt-in" : "amt-out"}">${fmt(tx.amount)}</td>
        `;
        txBody.appendChild(tr);
      });
    }

    function applyTableFilters() {
      const q = document.getElementById("searchInput").value.trim().toLowerCase();
      const t = document.getElementById("typeFilter").value;
      const filtered = currentTransactions.filter((tx) => {
        const matchesType = t === "all" || tx.direction === t;
        const matchesQuery = !q ||
          String(tx.description || "").toLowerCase().includes(q) ||
          String(tx.category || "").toLowerCase().includes(q) ||
          String(tx.date || "").toLowerCase().includes(q);
        return matchesType && matchesQuery;
      });
      renderTable(filtered);
    }

    function renderReport(data) {
      lastReportData = data;
      const cur = settings.currency || "₦";
      const s = data.summary || {};
      const inflow = Number(s.total_inflow || 0);
      const outflow = Number(s.total_outflow || 0);
      const net = Number(s.net_cash_flow || 0);

      document.getElementById("inflow").textContent = fmt(inflow, cur);
      document.getElementById("outflow").textContent = fmt(outflow, cur);
      document.getElementById("net").textContent = fmt(net, cur);
      document.getElementById("net").style.color = net >= 0 ? "var(--inflow)" : "var(--outflow)";
      document.getElementById("netMeta").textContent = net >= 0 ? "Net positive cash movement" : "Net negative cash movement";
      document.getElementById("txCountMeta").textContent = `${s.transaction_count || 0} records analyzed`;
      document.getElementById("flagged").textContent = s.flagged_items_count || 0;
      document.getElementById("flaggedMeta").textContent = `${fmt(s.flagged_total || 0, cur)} under review`;

      const maxFlow = Math.max(inflow, outflow, 1);
      document.getElementById("inflowBarLabel").textContent = fmt(inflow, cur);
      document.getElementById("outflowBarLabel").textContent = fmt(outflow, cur);
      document.getElementById("inflowBar").style.width = `${Math.min(100, (inflow / maxFlow) * 100)}%`;
      document.getElementById("outflowBar").style.width = `${Math.min(100, (outflow / maxFlow) * 100)}%`;

      const catList = document.getElementById("catList");
      catList.innerHTML = "";
      const cats = data.categories || [];
      if (!cats.length) {
        catList.innerHTML = `<div class="hint" style="margin:0;">No outflow categories found.</div>`;
      } else {
        const maxCat = Math.max(...cats.map((c) => Number(c.total_outflow || 0)), 1);
        cats.slice(0, 5).forEach((c) => {
          const pct = Math.min(100, (Number(c.total_outflow || 0) / maxCat) * 100);
          const div = document.createElement("div");
          div.className = "cat-item";
          div.innerHTML = `
            <div class="cat-meta">
              <span class="cat-name">${escapeHtml(c.category)}</span>
              <span class="cat-amt">${fmt(c.total_outflow, cur)}</span>
            </div>
            <div class="bar-track" style="height:7px;">
              <div class="bar-fill" style="width:${pct}%; background:var(--accent);"></div>
            </div>
          `;
          catList.appendChild(div);
        });
}

      const briefList = document.getElementById("briefList");
      briefList.innerHTML = "";
      const topCat = cats[0];
      const cashStatus = net >= 0 ? "positive" : "negative";
      const briefLines = [
        `Net cash flow is ${cashStatus} at ${fmt(net, cur)} (${fmt(inflow, cur)} inflow vs. ${fmt(outflow, cur)} outflow).`,
        topCat
          ? `Top outflow category is ${topCat.category} at ${fmt(topCat.total_outflow, cur)}.`
          : "No outflow categories recorded.",
        `Flagged ${s.flagged_items_count || 0} fee/duplicate item(s) totaling ${fmt(s.flagged_total || 0, cur)} for review.`
      ];
      briefLines.forEach((line) => {
        const li = document.createElement("li");
        li.textContent = line;
        briefList.appendChild(li);
      });

      const flagBox = document.getElementById("flagBox");
      const flagItems = document.getElementById("flagItems");
      const flags = data.flagged_items || [];
      if (flags.length) {
        flagBox.style.display = "block";
        flagItems.innerHTML = flags
          .map((f) => `<div>• <strong>${escapeHtml(f.date)}</strong> — ${escapeHtml(f.description)} (${fmt(f.amount, cur)}) <em>[${escapeHtml(f.reason)}]</em></div>`)
          .join("");
      } else {
        flagBox.style.display = "none";
        flagItems.innerHTML = "";
      }

      currentTransactions = data.transactions || [];
      document.getElementById("searchInput").value = "";
      document.getElementById("typeFilter").value = "all";
      renderTable(currentTransactions);
    }

    function pushHistory(sourceName, report) {
      const entry = {
        id: "h-" + Date.now(),
        source_name: sourceName,
        analyzed_at: new Date().toISOString().replace("T", " ").slice(0, 16) + " UTC",
        report
      };
      localHistory.unshift(entry);
      localHistory = localHistory.slice(0, 20);
      saveLocalHistory();
    }

    function renderHistory() {
      const listEl = document.getElementById("historyList");
      if (!localHistory.length) {
        listEl.innerHTML = `<div class="hint">No analyzed datasets yet. Run sample data or upload a file to build history.</div>`;
        return;
      }
      listEl.innerHTML = "";
      localHistory.forEach((item, idx) => {
        const sum = (item.report && item.report.summary) || {};
        const div = document.createElement("div");
        div.className = "history-item";
        div.innerHTML = `
          <div>
            <strong>${escapeHtml(item.source_name)}</strong>
            <div class="hint" style="margin:2px 0 0;">${escapeHtml(item.analyzed_at)} · ${sum.transaction_count || 0} records · Net ${fmt(sum.net_cash_flow || 0)}</div>
          </div>
          <button type="button" class="ghost" data-hidx="${idx}">Load Snapshot</button>
        `;
        div.querySelector("button").addEventListener("click", () => {
          renderReport(item.report);
          switchView("analyzeView");
          showToast("Loaded snapshot from history");
        });
        listEl.appendChild(div);
      });
    }

    document.getElementById("clearHistoryBtn").addEventListener("click", async () => {
      localHistory = [];
      saveLocalHistory();
      try { await fetch("/api/history", { method: "DELETE" }); } catch (_) {}
      renderHistory();
      showToast("History cleared");
    });

    document.getElementById("exportCsvBtn").addEventListener("click", () => {
      if (!currentTransactions.length) {
        showToast("No records to export yet");
        return;
      }
      const headers = ["date", "description", "category", "direction", "amount"];
      const escape = (v) => `"${String(v ?? "").replace(/"/g, '""')}"`;
      const lines = [
        headers.join(","),
        ...currentTransactions.map((tx) => headers.map((h) => escape(tx[h])).join(","))
      ];
      const blob = new Blob([lines.join("\\n")], { type: "text/csv;charset=utf-8;" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = "kevsight-normalized-records.csv";
      a.click();
      URL.revokeObjectURL(url);
      showToast("Exported CSV");
    });

    const dropzone = document.getElementById("dropzone");
    const fileInput = document.getElementById("fileInput");
    const fileLabel = document.getElementById("fileLabel");
    function showSelectedFile(file) {
      if (!file) return;
      fileLabel.textContent = file.name;
      dropzone.classList.remove("dragover");
    }
    dropzone.addEventListener("click", (event) => {
      if (event.target !== fileInput) fileInput.click();
    });
    dropzone.addEventListener("keydown", (event) => {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        fileInput.click();
      }
    });
    fileInput.addEventListener("change", () => showSelectedFile(fileInput.files[0]));
    ["dragenter", "dragover"].forEach((name) => dropzone.addEventListener(name, (event) => {
      event.preventDefault();
      dropzone.classList.add("dragover");
    }));
    ["dragleave", "drop"].forEach((name) => dropzone.addEventListener(name, (event) => {
      event.preventDefault();
      dropzone.classList.remove("dragover");
    }));
    dropzone.addEventListener("drop", (event) => {
      const file = event.dataTransfer.files && event.dataTransfer.files[0];
      if (!file) return;
      const allowed = /\\.(csv|xlsx|xls|pdf)$/i.test(file.name);
      if (!allowed) {
        showToast("Please choose a CSV, Excel or PDF file");
        return;
      }
      const transfer = new DataTransfer();
      transfer.items.add(file);
      fileInput.files = transfer.files;
      showSelectedFile(file);
    });

    document.getElementById("searchInput").addEventListener("input", applyTableFilters);
    document.getElementById("typeFilter").addEventListener("change", applyTableFilters);

    document.getElementById("sampleBtn").addEventListener("click", async () => {
      try {
        const params = new URLSearchParams({
          currency: settings.currency || "₦",
          check_duplicates: String(Boolean(settings.checkDuplicates))
        });
        const res = await fetch(`/api/sample?${params.toString()}`);
        if (!res.ok) throw new Error("Sample request failed");
        const data = await res.json();
        pushHistory("Sample Dataset", data);
        renderReport(data);
        switchView("analyzeView");
        showToast("Sample data loaded");
      } catch (_) {
        showToast("Could not load sample data. Please try again.");
      }
    });

    document.getElementById("uploadForm").addEventListener("submit", async (e) => {
      e.preventDefault();
      const input = document.getElementById("fileInput");
      if (!input.files.length) return;
      const file = input.files[0];
      const btn = document.getElementById("analyzeBtn");
      btn.textContent = "Analyzing...";
      btn.disabled = true;
      try {
        const params = new URLSearchParams({
          currency: settings.currency || "₦",
          check_duplicates: String(Boolean(settings.checkDuplicates))
        });
        const formData = new FormData();
        formData.append("file", file);
        const res = await fetch(`/api/analyze?${params.toString()}`, { method: "POST", body: formData });
        const data = await res.json().catch(() => ({}));
        if (!res.ok) {
          showToast(data.detail || "Could not process this file");
          return;
        }
        pushHistory(file.name, data);
        renderReport(data);
        showToast(`Analysis complete: ${file.name}`);
      } catch (error) {
        showToast("Could not connect to Kevsight. Check your connection and try again.");
      } finally {
        btn.textContent = "Analyze Data";
        btn.disabled = false;
      }
    });

    document.getElementById("healthCheckBtn").addEventListener("click", async () => {
      const out = document.getElementById("healthResult");
      out.textContent = "Checking service...";
      try {
        const res = await fetch("/health");
        const data = await res.json();
        out.textContent = res.ok ? `Service healthy (${data.service || "kevsight"})` : "Service returned an unexpected status.";
      } catch (_) {
        out.textContent = "Could not reach /health endpoint.";
      }
    });

    applyTheme(settings.theme);
    syncSettingsInputs();
  </script>
</body>
</html>
"""


@app.get("/api/sample")
async def sample_analysis(currency: str = "₦", check_duplicates: bool = True) -> JSONResponse:
    report = analyze_transactions(SAMPLE_TRANSACTIONS, currency=currency, check_duplicates=check_duplicates)
    record_history("Sample Dataset", report)
    return JSONResponse(report)


@app.post("/api/analyze")
async def analyze_upload(
    file: UploadFile = File(...),
    currency: str = "₦",
    check_duplicates: bool = True,
) -> JSONResponse:
    filename = file.filename or "upload.csv"
    allowed_extensions = {".csv", ".xlsx", ".xls", ".pdf"}
    extension = os.path.splitext(filename.lower())[1]
    if extension not in allowed_extensions:
        raise HTTPException(status_code=400, detail="Unsupported file type. Please upload CSV, Excel, or PDF.")

    content = await file.read()
    max_bytes = 15 * 1024 * 1024
    if len(content) > max_bytes:
        raise HTTPException(status_code=413, detail="File is too large. Maximum upload size is 15 MB.")
    if not content:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")

    try:
        transactions = parse_uploaded_file(filename, content)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Could not read this file. Check that it is valid and not password-protected. ({type(exc).__name__})")

    if not transactions:
        raise HTTPException(
            status_code=422,
            detail="No usable transactions were detected. Check that the file contains a date, description, and amount (or debit/credit columns)."
        )

    report = analyze_transactions(transactions, currency=currency, check_duplicates=check_duplicates)
    record_history(filename, report)
    return JSONResponse(report)
