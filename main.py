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
