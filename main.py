import io
import os
import re
from typing import Any, Dict, List, Optional

import pandas as pd
import pdfplumber
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse

app = FastAPI(title="Kevsight MVP")


FEE_KEYWORDS = [
    "fee", "charge", "vat", "sms alert", "maintenance", "commission",
    "stamp duty", "levy", "penalty", "overdraft", "transfer fee", "service charge"
]

CATEGORY_RULES = {
    "Fees & Bank Charges": FEE_KEYWORDS,
    "Software & Subscriptions": ["aws", "google", "canva", "zoom", "slack", "notion", "openai", "anthropic", "github", "vercel", "render", "subscription"],
    "Utilities & Internet": ["electric", "power", "water", "internet", "fiber", "data", "airtime", "utility"],
    "Rent & Operations": ["rent", "lease", "office", "workspace", "storage"],
    "Payroll & Contractors": ["salary", "payroll", "wages", "contractor", "retainer", "payout"],
    "Sales & Revenue": ["invoice", "payment received", "settlement", "deposit", "sales", "stripe", "paystack", "flutterwave", "client"],
    "Supplies & Inventory": ["inventory", "stock", "supplies", "wholesale", "vendor", "supplier"]
}


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

    date_col = next((c for c in df.columns if any(k in c for k in ["date", "time", "posted", "value"])), None)
    desc_col = next((c for c in df.columns if any(k in c for k in ["desc", "narration", "details", "memo", "particulars", "merchant", "name"])), None)
    amount_col = next((c for c in df.columns if c in {"amount", "amt", "value"} or "amount" in c), None)
    debit_col = next((c for c in df.columns if any(k in c for k in ["debit", "withdrawal", "outflow", "dr"])), None)
    credit_col = next((c for c in df.columns if any(k in c for k in ["credit", "deposit", "inflow", "cr"])), None)

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
    date_pattern = re.compile(r"^(\d{4}-\d{2}-\d{2}|\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\w{3}\s+\d{1,2},?\s*\d{0,4})\s+(.+?)\s+(-?\(?\$?[\d,]+\.\d{2}\)?)$")

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


def analyze_transactions(transactions: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not transactions:
        return {
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
            "plain_english_brief": ["No valid transactions were found in the uploaded file."],
            "transactions": [],
        }

    df = pd.DataFrame(transactions)
    inflow = float(df[df["direction"] == "inflow"]["amount"].sum()) if "inflow" in df["direction"].values else 0.0
    outflow = float(df[df["direction"] == "outflow"]["amount"].sum()) if "outflow" in df["direction"].values else 0.0
    net_cash = round(inflow - outflow, 2)

    flagged: List[Dict[str, Any]] = []
    for _, row in df.iterrows():
        desc_l = str(row["description"]).lower()
        reasons = []
        if row["category"] == "Fees & Bank Charges" or any(k in desc_l for k in FEE_KEYWORDS):
            reasons.append("Bank/service fee")
        if reasons:
            flagged.append({
                "date": row["date"],
                "description": row["description"],
                "amount": row["amount"],
                "reason": ", ".join(reasons),
            })

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
        f"Top outflow category is {categories[0]['category']} at ${categories[0]['total_outflow']:,.2f}."
        if categories
        else "No outflow categories recorded."
    )
    cash_status = "positive" if net_cash >= 0 else "negative"

    brief = [
        f"Net cash flow is {cash_status} at ${net_cash:,.2f} (${inflow:,.2f} inflow vs. ${outflow:,.2f} outflow).",
        top_cat_text,
        f"Flagged {len(flagged)} fee/duplicate item(s) totaling ${flagged_total:,.2f} for review.",
    ]

    return {
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


@app.get("/", response_class=HTMLResponse)
async def home() -> str:
    return """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Kevsight — Finance Cash Flow & Leak Snapshot</title>
  <style>
    body { font-family: system-ui, -apple-system, sans-serif; margin: 0; background: #f8f7fb; color: #18181b; }
    .container { max-width: 1020px; margin: 32px auto; padding: 0 20px; }
    header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 24px; }
    .brand { font-size: 24px; font-weight: 800; color: #4c1d95; }
    button { background: #4c1d95; color: #fff; border: none; padding: 10px 16px; border-radius: 8px; cursor: pointer; font-weight: 600; }
    .card { background: #fff; border: 1px solid #e4e4e7; border-radius: 12px; padding: 18px; margin-bottom: 18px; }
    .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 14px; }
    .kpi-label { font-size: 12px; color: #52525b; text-transform: uppercase; }
    .kpi-val { font-size: 24px; font-weight: 700; margin-top: 6px; }
    table { width: 100%; border-collapse: collapse; font-size: 14px; }
    th, td { text-align: left; padding: 10px; border-bottom: 1px solid #f4f4f5; }
    th { color: #52525b; font-size: 12px; text-transform: uppercase; }
    .inflow { color: #15803d; font-weight: 600; }
    .outflow { color: #b91c1c; font-weight: 600; }
  </style>
</head>
<body>
  <div class="container">
    <header>
      <div class="brand">Kevsight</div>
      <button id="sampleBtn">Try Sample Data</button>
    </header>

    <div class="card">
      <form id="uploadForm" style="display:flex; gap:12px; flex-wrap:wrap; align-items:center;">
        <input type="file" id="fileInput" accept=".csv,.xlsx,.xls,.pdf" required />
        <button type="submit">Analyze Statement</button>
      </form>
    </div>

    <div class="grid" id="kpis">
      <div class="card"><div class="kpi-label">Total Inflow</div><div class="kpi-val" id="inflow">$0.00</div></div>
      <div class="card"><div class="kpi-label">Total Outflow</div><div class="kpi-val" id="outflow">$0.00</div></div>
      <div class="card"><div class="kpi-label">Net Cash Flow</div><div class="kpi-val" id="net">$0.00</div></div>
      <div class="card"><div class="kpi-label">Flagged Items</div><div class="kpi-val" id="flagged">0</div></div>
    </div>

    <div class="card">
      <h3 style="margin-top:0;">Plain-English Brief</h3>
      <ul id="briefList"><li>Click "Try Sample Data" or upload a CSV, Excel, or PDF statement to begin.</li></ul>
    </div>

    <div class="card">
      <h3 style="margin-top:0;">Normalized Transactions</h3>
      <table>
        <thead>
          <tr><th>Date</th><th>Description</th><th>Category</th><th>Type</th><th>Amount</th></tr>
        </thead>
        <tbody id="txBody"></tbody>
      </table>
    </div>
  </div>

  <script>
    const fmt = (n) => "$" + Number(n || 0).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 });

    function renderReport(data) {
      document.getElementById("inflow").textContent = fmt(data.summary.total_inflow);
      document.getElementById("outflow").textContent = fmt(data.summary.total_outflow);
      document.getElementById("net").textContent = fmt(data.summary.net_cash_flow);
      document.getElementById("flagged").textContent = data.summary.flagged_items_count;

      const briefList = document.getElementById("briefList");
      briefList.innerHTML = "";
      (data.plain_english_brief || []).forEach((line) => {
        const li = document.createElement("li");
        li.textContent = line;
        briefList.appendChild(li);
      });

      const txBody = document.getElementById("txBody");
      txBody.innerHTML = "";
      (data.transactions || []).forEach((tx) => {
        const tr = document.createElement("tr");
        tr.innerHTML = `
          <td>${tx.date}</td>
          <td>${tx.description}</td>
          <td>${tx.category}</td>
          <td class="${tx.direction}">${tx.direction}</td>
          <td class="${tx.direction}">${fmt(tx.amount)}</td>
        `;
        txBody.appendChild(tr);
      });
    }

    document.getElementById("sampleBtn").addEventListener("click", async () => {
      const res = await fetch("/api/sample");
      const data = await res.json();
      renderReport(data);
    });

    document.getElementById("uploadForm").addEventListener("submit", async (e) => {
      e.preventDefault();
      const input = document.getElementById("fileInput");
      if (!input.files.length) return;
      const formData = new FormData();
      formData.append("file", input.files[0]);
      const res = await fetch("/api/analyze", { method: "POST", body: formData });
      const data = await res.json();
      if (!res.ok) {
        alert(data.detail || "Could not process file.");
        return;
      }
      renderReport(data);
    });
  </script>
</body>
</html>
"""


@app.get("/api/sample")
async def sample_analysis() -> JSONResponse:
    return JSONResponse(analyze_transactions(SAMPLE_TRANSACTIONS))


@app.post("/api/analyze")
async def analyze_upload(file: UploadFile = File(...)) -> JSONResponse:
    content = await file.read()
    transactions = parse_uploaded_file(file.filename or "upload.csv", content)
    return JSONResponse(analyze_transactions(transactions))
