import io
import os
import re
import hashlib
import hmac
import secrets
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import pandas as pd
import pdfplumber
from fastapi import FastAPI, File, HTTPException, UploadFile, Request
from fastapi.responses import HTMLResponse, JSONResponse
from starlette.middleware.sessions import SessionMiddleware
from sqlalchemy import create_engine, String, Integer, DateTime, ForeignKey, JSON, Text, select, desc
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker
from pydantic import BaseModel

app = FastAPI(title="Kevsight")

# Configure DATABASE_URL in your local environment. Render production should use
# a managed PostgreSQL database and its Internal Database URL.
DATABASE_URL = os.getenv("DATABASE_URL", "").strip()
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql+psycopg://", 1)
elif DATABASE_URL.startswith("postgresql://"):
    DATABASE_URL = DATABASE_URL.replace("postgresql://", "postgresql+psycopg://", 1)

engine = create_engine(DATABASE_URL, pool_pre_ping=True) if DATABASE_URL else None
SessionLocal = sessionmaker(bind=engine, expire_on_commit=False) if engine else None

class Base(DeclarativeBase):
    pass

class User(Base):
    __tablename__ = "users"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    email: Mapped[str] = mapped_column(String(320), unique=True, index=True)
    password_hash: Mapped[str] = mapped_column(String(256))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

class AnalysisRecord(Base):
    __tablename__ = "analysis_records"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    source_name: Mapped[str] = mapped_column(String(255))
    analyzed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    report: Mapped[dict] = mapped_column(JSON)

app.add_middleware(
    SessionMiddleware,
    secret_key=os.getenv("SESSION_SECRET", "local-development-secret-change-this-before-deployment"),
    session_cookie="kevsight_session",
    max_age=60 * 60 * 24 * 7,
    same_site="lax",
    https_only=os.getenv("COOKIE_SECURE", "false").lower() == "true",
)

@app.on_event("startup")
async def initialise_database() -> None:
    if engine is not None:
        Base.metadata.create_all(bind=engine)


def get_db():
    if SessionLocal is None:
        raise HTTPException(status_code=503, detail="Database is not configured. Set DATABASE_URL to your PostgreSQL connection string.")
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def password_digest(password: str, salt: bytes | None = None) -> str:
    salt = salt or secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 310_000)
    return f"pbkdf2_sha256${salt.hex()}${digest.hex()}"


def verify_password(password: str, stored: str) -> bool:
    try:
        scheme, salt_hex, digest_hex = stored.split("$", 2)
        if scheme != "pbkdf2_sha256":
            return False
        candidate = password_digest(password, bytes.fromhex(salt_hex)).split("$")[2]
        return hmac.compare_digest(candidate, digest_hex)
    except (ValueError, TypeError):
        return False


def authenticated_user(request: Request, db):
    user_id = request.session.get("user_id")
    if not user_id:
        raise HTTPException(status_code=401, detail="Please sign in to continue.")
    user = db.get(User, int(user_id))
    if user is None:
        request.session.clear()
        raise HTTPException(status_code=401, detail="Your session has expired. Please sign in again.")
    return user

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
        else "No outflow categories were identified in this dataset."
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


def record_history(db, user_id: int, source_name: str, report: Dict[str, Any]) -> Dict[str, Any]:
    row = AnalysisRecord(user_id=user_id, source_name=source_name, report=report)
    db.add(row)
    db.commit()
    db.refresh(row)
    return {
        "id": str(row.id),
        "source_name": row.source_name,
        "analyzed_at": row.analyzed_at.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "report": row.report,
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


@app.get("/health")
async def health() -> JSONResponse:
    return JSONResponse({"status": "ok", "service": "kevsight"})


@app.post("/api/auth/register")
async def register(request: Request):
    data = await request.json()
    email = str(data.get("email", "")).strip().lower()
    password = str(data.get("password", ""))
    if not re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", email):
        raise HTTPException(status_code=400, detail="Enter a valid email address.")
    if len(password) < 10 or len(password) > 128:
        raise HTTPException(status_code=400, detail="Password must be between 10 and 128 characters.")
    db_gen = get_db()
    db = next(db_gen)
    try:
        existing = db.scalar(select(User).where(User.email == email))
        if existing:
            raise HTTPException(status_code=409, detail="An account with this email already exists.")
        user = User(email=email, password_hash=password_digest(password))
        db.add(user)
        db.commit()
        db.refresh(user)
        request.session.clear()
        request.session["user_id"] = user.id
        return {"ok": True, "user": {"email": user.email}}
    finally:
        db.close()


@app.post("/api/auth/login")
async def login(request: Request):
    data = await request.json()
    email = str(data.get("email", "")).strip().lower()
    password = str(data.get("password", ""))
    db_gen = get_db()
    db = next(db_gen)
    try:
        user = db.scalar(select(User).where(User.email == email))
        if user is None or not verify_password(password, user.password_hash):
            raise HTTPException(status_code=401, detail="Email or password is incorrect.")
        request.session.clear()
        request.session["user_id"] = user.id
        return {"ok": True, "user": {"email": user.email}}
    finally:
        db.close()


@app.get("/api/auth/me")
async def who_am_i(request: Request):
    db_gen = get_db()
    db = next(db_gen)
    try:
        user = authenticated_user(request, db)
        return {"authenticated": True, "user": {"email": user.email}}
    finally:
        db.close()


@app.post("/api/auth/logout")
async def logout(request: Request):
    request.session.clear()
    return {"ok": True}


@app.get("/api/history")
async def get_history(request: Request):
    db_gen = get_db()
    db = next(db_gen)
    try:
        user = authenticated_user(request, db)
        rows = db.scalars(select(AnalysisRecord).where(AnalysisRecord.user_id == user.id).order_by(desc(AnalysisRecord.analyzed_at)).limit(MAX_HISTORY)).all()
        return {"items": [{"id": str(r.id), "source_name": r.source_name, "analyzed_at": r.analyzed_at.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"), "report": r.report} for r in rows]}
    finally:
        db.close()


@app.delete("/api/history")
async def clear_history(request: Request):
    db_gen = get_db()
    db = next(db_gen)
    try:
        user = authenticated_user(request, db)
        db.query(AnalysisRecord).filter(AnalysisRecord.user_id == user.id).delete(synchronize_session=False)
        db.commit()
        return {"ok": True}
    finally:
        db.close()


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

    .landing{max-width:1180px;margin:0 auto;padding:0 28px;color:#17152a}.landing-nav{height:82px;display:flex;align-items:center;justify-content:space-between;border-bottom:1px solid #e9e6f1}.landing-brand{display:flex;align-items:center;gap:11px;color:#19152d;text-decoration:none;font-weight:850;font-size:21px}.landing-brand .brand-mark{display:grid;place-items:center;width:38px;height:38px;border-radius:12px;background:#5b21b6;color:white}.landing-nav-note{font-size:13px;color:#777187}.landing-signin{background:white!important;color:#5120a5!important;border:1px solid #ded6ed!important;padding:10px 20px!important}.hero{display:grid;grid-template-columns:1.05fr .95fr;gap:65px;align-items:center;padding:74px 0 82px}.hero-copy .eyebrow,.landing-features small,.auth-intro>small{font-size:11px;font-weight:850;letter-spacing:.15em;color:#6b36bb}.hero h1{font-size:clamp(42px,5.3vw,66px);line-height:1.06;letter-spacing:-.055em;margin:19px 0;color:#19152d}.hero h1 em{font-style:normal;color:#6936b7}.hero-copy>p{font-size:16px;line-height:1.8;color:#69647a;max-width:550px}.hero-actions{display:flex;align-items:center;gap:16px;flex-wrap:wrap;margin:28px 0 20px}.hero-actions button,.auth-submit{background:#5b21b6;color:white;border:0;border-radius:10px;padding:14px 22px;font-weight:750}.hero-actions span{font-size:11px;color:#817b90}.hero-proof{display:flex;gap:19px;flex-wrap:wrap;color:#615b70;font-size:12px}.preview-card{background:#fff;border:1px solid #e9e5f0;border-radius:20px;padding:25px;box-shadow:0 22px 65px #33204d12;transform:rotate(1deg)}.preview-top{display:flex;justify-content:space-between;align-items:center}.preview-top small,.preview-kpis small,.chart-title{font-size:10px;letter-spacing:.08em;color:#817a91;font-weight:750}.preview-top h3{margin:7px 0 0;font-size:19px}.live-pill{font-size:9px;color:#16804b;background:#e8f7ee;padding:7px 9px;border-radius:20px}.preview-kpis{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin:24px 0}.preview-kpis>div{background:#f8f6fb;border-radius:12px;padding:15px}.preview-kpis strong{display:block;font-size:25px;margin:8px 0 4px;letter-spacing:-.04em}.preview-kpis span{font-size:10px;color:#817a91}.positive{color:#16804b}.chart-title{display:flex;justify-content:space-between;margin:20px 0 12px}.chart-title small{font-weight:500;letter-spacing:0}.mini-chart{height:145px;display:flex;gap:13px;border-bottom:1px solid #eae6f0;background:repeating-linear-gradient(to bottom,transparent 0,transparent 35px,#f0edf5 36px)}.chart-y{display:flex;flex-direction:column;justify-content:space-between;color:#9992a6;font-size:9px;padding-bottom:5px}.chart-bars{flex:1;display:flex;align-items:end;justify-content:space-around;gap:10px;padding:0 5px}.chart-bars i{display:block;flex:1;max-width:30px;border-radius:5px 5px 0 0;background:linear-gradient(180deg,#9564d8,#5b21b6)}.chart-months{display:flex;justify-content:space-around;margin:8px 0;color:#8c8599;font-size:9px}.preview-foot{display:flex;justify-content:space-between;border-top:1px solid #eeeaf3;padding-top:15px;margin-top:17px;color:#696278;font-size:10px}.landing-features{padding:60px 0;border-top:1px solid #eae6f0}.landing-features h2,.auth-intro h2{font-size:34px;letter-spacing:-.04em;margin:12px 0 28px}.feature-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:18px}.feature-grid article{background:#fff;border:1px solid #ebe7f1;border-radius:15px;padding:23px}.feature-grid b{color:#7949bd;font-size:12px}.feature-grid h3{margin:14px 0 8px}.feature-grid p{font-size:13px;line-height:1.7;color:#777187;margin:0}.auth-section{display:grid;grid-template-columns:1fr 430px;gap:60px;align-items:center;padding:75px 8%;margin:25px 0;background:#f2eef9;border-radius:22px}.auth-intro h2{line-height:1.15}.auth-intro p{color:#716b7e;line-height:1.7;max-width:390px}.auth-assurance{font-size:11px;color:#655d74;margin-top:24px}.auth-card{background:#fff;padding:27px;border:1px solid #e7e1ef;border-radius:17px;box-shadow:0 12px 35px #24133c0b}.auth-tabs{display:flex;gap:8px;border-bottom:1px solid #eeeaf3;margin-bottom:22px}.auth-tab{background:transparent!important;color:#81798e!important;border:0!important;border-bottom:2px solid transparent!important;border-radius:0!important;padding:10px 13px!important}.auth-tab.active{color:#5722ad!important;border-bottom-color:#5722ad!important}.auth-card label{display:block;font-size:12px;font-weight:700;margin:14px 0 7px}.auth-card input{width:100%;height:44px;border:1px solid #e2ddea;border-radius:9px;padding:0 12px;font:inherit}.auth-hint{font-size:11px;color:#80798e;line-height:1.5}.auth-error{font-size:12px;color:#b42318;min-height:16px}.auth-submit{width:100%;margin-top:5px}.landing-footer{display:flex;justify-content:space-between;padding:22px 0 30px;color:#888195;font-size:11px}.container header .header-actions{display:flex;gap:8px;flex-wrap:wrap}
    @media(max-width:850px){.hero{grid-template-columns:1fr;gap:36px;padding:55px 0}.preview-card{max-width:600px;width:100%;transform:none}.auth-section{grid-template-columns:1fr;gap:25px;padding:38px 25px}.auth-card{max-width:500px;width:100%}}
    @media(max-width:600px){.landing{padding:0 16px}.landing-nav{height:68px}.landing-nav-note{display:none}.hero h1{font-size:43px}.feature-grid{grid-template-columns:1fr}.landing-features h2,.auth-intro h2{font-size:28px}.preview-card{padding:17px}.preview-kpis strong{font-size:21px}.landing-footer{gap:12px;flex-wrap:wrap}.auth-section{padding:30px 17px}}

  </style>
</head>
<body>
  <main class="landing" id="landingPage">
    <nav class="landing-nav"><a class="landing-brand" href="#top"><span class="brand-mark">K</span><span>Kevsight</span></a><span class="landing-nav-note">Financial clarity for growing businesses</span><button class="landing-signin" id="navSignIn">Sign in</button></nav>
    <section class="hero" id="top"><div class="hero-copy"><div class="eyebrow">BUSINESS INTELLIGENCE, MADE CLEAR</div><h1>See your numbers.<br><em>Understand your business.</em></h1><p>Bring sales and financial data into one clear workspace. Track performance, understand cash flow and turn business records into practical insights.</p><div class="hero-actions"><button id="heroGetStarted">Create free account</button><span>Simple email registration · Your own private workspace</span></div><div class="hero-proof"><span>✓ Sales performance</span><span>✓ Cash-flow visibility</span><span>✓ Financial insights</span></div></div>
      <div class="preview-card"><div class="preview-top"><div><small>WORKSPACE OVERVIEW</small><h3>Business snapshot</h3></div><span class="live-pill">● SAMPLE VIEW</span></div><div class="preview-kpis"><div><small>Total sales</small><strong>₦8.42m</strong><span>Across selected period</span></div><div><small>Net cash flow</small><strong class="positive">₦2.16m</strong><span>Illustrative data</span></div></div><div class="chart-title">Monthly performance <small>Illustrative</small></div><div class="mini-chart"><div class="chart-y"><span>10m</span><span>5m</span><span>0</span></div><div class="chart-bars"><i style="height:38%"></i><i style="height:52%"></i><i style="height:44%"></i><i style="height:69%"></i><i style="height:61%"></i><i style="height:84%"></i><i style="height:74%"></i><i style="height:96%"></i></div></div><div class="chart-months"><span>Jan</span><span>Feb</span><span>Mar</span><span>Apr</span><span>May</span><span>Jun</span><span>Jul</span><span>Aug</span></div><div class="preview-foot"><span>Sales trends</span><span>Category insights</span><span>Financial reports</span></div></div></section>
    <section class="landing-features"><div><small>ONE WORKSPACE</small><h2>From raw records to confident decisions.</h2></div><div class="feature-grid"><article><b>01</b><h3>Sales analytics</h3><p>Explore revenue, product performance, customer patterns and trends from your business datasets.</p></article><article><b>02</b><h3>Financial statement insights</h3><p>Build a clearer view of profitability, financial position, cash flows and key ratios from company reports.</p></article><article><b>03</b><h3>Business dashboards</h3><p>Bring important metrics, charts and findings together in a practical management view.</p></article></div></section>
    <section class="auth-section" id="authSection"><div class="auth-intro"><small>GET STARTED WITH KEVSIGHT</small><h2>Your business numbers,<br>in one place.</h2><p>Create an account to keep your analysis history connected to your private workspace.</p><div class="auth-assurance">Your account is protected with password hashing and a secure session cookie.</div></div><div class="auth-card"><div class="auth-tabs"><button class="auth-tab active" id="loginTab">Sign in</button><button class="auth-tab" id="registerTab">Create account</button></div><form id="authForm"><label for="authEmail">Email address</label><input id="authEmail" type="email" autocomplete="email" placeholder="you@company.com" required><label for="authPassword">Password</label><input id="authPassword" type="password" autocomplete="current-password" minlength="10" placeholder="At least 10 characters" required><p class="auth-hint" id="authHint">Welcome back. Sign in to open your workspace.</p><p class="auth-error" id="authError" role="alert"></p><button class="auth-submit" type="submit" id="authSubmit">Sign in to Kevsight</button></form></div></section>
    <footer class="landing-footer"><span>© 2026 Kevsight</span><span>Finance &amp; Sales Data Intelligence</span></footer>
  </main>

  <div class="container" id="appShell" style="display:none;">
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
        <button type="button" class="ghost" id="logoutBtn">Sign out</button>
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
            <h2 class="section-title">Key Findings</h2>
          </div>
          <ul class="brief-list" id="briefList">
            <li>Your analysis summary will appear here after you load sample data or analyse a file.</li>
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
          <div class="hint" style="margin:0;">Yes. Positive amounts or credit rows are treated as inflows/sales, while outflows are grouped into categories, checked for fees/duplicates, and included in the analysis summary.</div>
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
    let authMode = "login";
    const landingPage = document.getElementById("landingPage");
    const appShell = document.getElementById("appShell");
    function setAuthMode(mode) {
      authMode = mode;
      document.getElementById("loginTab").classList.toggle("active", mode === "login");
      document.getElementById("registerTab").classList.toggle("active", mode === "register");
      document.getElementById("authSubmit").textContent = mode === "login" ? "Sign in to Kevsight" : "Create my account";
      document.getElementById("authHint").textContent = mode === "login" ? "Welcome back. Sign in to open your workspace." : "Use a valid email and a password of at least 10 characters.";
      document.getElementById("authPassword").autocomplete = mode === "login" ? "current-password" : "new-password";
      document.getElementById("authError").textContent = "";
    }
    function showWorkspace(email) {
      landingPage.style.display = "none";
      appShell.style.display = "block";
      const brand = document.getElementById("brandDisplay");
      if (brand) brand.textContent = "Kevsight";
      const sub = document.querySelector(".brand-sub");
      if (sub) sub.textContent = email;
      loadHistoryFromServer();
    }
    async function checkSession() {
      try {
        const res = await fetch("/api/auth/me");
        if (res.ok) { const data = await res.json(); showWorkspace(data.user.email); }
      } catch (_) {}
    }
    document.getElementById("loginTab").addEventListener("click", () => setAuthMode("login"));
    document.getElementById("registerTab").addEventListener("click", () => setAuthMode("register"));
    document.getElementById("navSignIn").addEventListener("click", () => document.getElementById("authSection").scrollIntoView({behavior:"smooth"}));
    document.getElementById("heroGetStarted").addEventListener("click", () => { setAuthMode("register"); document.getElementById("authSection").scrollIntoView({behavior:"smooth"}); });
    document.getElementById("authForm").addEventListener("submit", async (event) => {
      event.preventDefault();
      const button = document.getElementById("authSubmit");
      const error = document.getElementById("authError");
      button.disabled = true; button.textContent = authMode === "login" ? "Signing in..." : "Creating account..."; error.textContent = "";
      try {
        const response = await fetch(`/api/auth/${authMode === "login" ? "login" : "register"}`, {method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify({email:document.getElementById("authEmail").value, password:document.getElementById("authPassword").value})});
        const data = await response.json();
        if (!response.ok) throw new Error(data.detail || "Could not complete your request.");
        showWorkspace(data.user.email);
      } catch (err) { error.textContent = err.message || "Unable to connect. Please try again."; }
      finally { button.disabled = false; button.textContent = authMode === "login" ? "Sign in to Kevsight" : "Create my account"; }
    });
    document.getElementById("logoutBtn").addEventListener("click", async () => {
      await fetch("/api/auth/logout", {method:"POST"});
      location.reload();
    });
    checkSession();

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

    async function loadHistoryFromServer() {
      try {
        const response = await fetch("/api/history");
        if (!response.ok) return;
        const data = await response.json();
        localHistory = Array.isArray(data.items) ? data.items : [];
        renderHistory();
      } catch (_) {}
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
        `Net cash flow: ${fmt(net, cur)}. Total inflows: ${fmt(inflow, cur)}; total outflows: ${fmt(outflow, cur)}.`,
        topCat
          ? `Largest outflow category: ${topCat.category} (${fmt(topCat.total_outflow, cur)}).`
          : "No outflow categories were identified in this dataset.",
        `Items requiring review: ${s.flagged_items_count || 0}, with a combined value of ${fmt(s.flagged_total || 0, cur)}.`
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
async def sample_analysis(request: Request, currency: str = "₦", check_duplicates: bool = True) -> JSONResponse:
    db_gen = get_db()
    db = next(db_gen)
    try:
        user = authenticated_user(request, db)
        report = analyze_transactions(SAMPLE_TRANSACTIONS, currency=currency, check_duplicates=check_duplicates)
        record_history(db, user.id, "Sample Dataset", report)
        return JSONResponse(report)
    finally:
        db.close()


@app.post("/api/analyze")
async def analyze_upload(
    request: Request,
    file: UploadFile = File(...),
    currency: str = "₦",
    check_duplicates: bool = True,
) -> JSONResponse:
    db_gen = get_db()
    db = next(db_gen)
    user = authenticated_user(request, db)
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
    record_history(db, user.id, filename, report)
    db.close()
    return JSONResponse(report)
