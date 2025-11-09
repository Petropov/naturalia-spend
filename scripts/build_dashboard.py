# scripts/build_dashboard.py
import os
import re
import pandas as pd

RCPT = "data/receipts.csv"
ITEM = "data/items.csv"
OUT_DIR = "artifacts"
OUT_HTML = os.path.join(OUT_DIR, "dashboard.html")
LEARNED_CATS = "artifacts/categories_learned.csv"  # optional: produced by scripts/learn_categories.py

os.makedirs(OUT_DIR, exist_ok=True)

# --------- helpers ---------
def read_csv_safe(path, **kw) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_csv(path, **kw)

def fmt_eur(x) -> str:
    try:
        val = float(x) if pd.notna(x) else 0.0
        return f"€{val:,.2f}".replace(",", " ")
    except Exception:
        return "€0.00"

def table(df: pd.DataFrame, cols, header=None, empty_msg="No data") -> str:
    if df is None or df.empty:
        return f"<em>{empty_msg}</em>"
    use = [c for c in cols if c in df.columns]
    if not use:
        return f"<em>{empty_msg}</em>"
    t = df.loc[:, use].copy()
    # format money-looking columns
    for c in t.columns:
        if any(k in c for k in ("total", "price", "spend")) or c in ("items_sum",):
            t[c] = t[c].apply(fmt_eur)
    if header:
        t.columns = header
    return t.to_html(index=False, border=0, classes="tbl")

def normalize_for_frequency(s: str) -> str:
    s = (s or "").lower()
    # strip pack sizes like "4x125g", "500 ml", "2×", etc.
    s = re.sub(r"\b\d+\s*(x|×)?\s*\d*\s*(ml|l|g|kg|cl|pack|pcs)\b", " ", s)
    # strip standalone sizes like "500g"
    s = re.sub(r"\b\d+\s*(ml|l|g|kg|cl)\b", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

# --------- load & normalize ----------
receipts = read_csv_safe(RCPT)
items    = read_csv_safe(ITEM)

# receipts: dates & numbers
if not receipts.empty and "date" in receipts.columns:
    receipts["date"] = pd.to_datetime(receipts["date"], errors="coerce")
else:
    receipts["date"] = pd.NaT

for c in ("total", "subtotal", "tax", "confidence"):
    if c in receipts.columns:
        receipts[c] = pd.to_numeric(receipts[c], errors="coerce")

# items: numbers + normalized name
if not items.empty:
    for c in ("qty", "unit_price", "line_total"):
        if c in items.columns:
            items[c] = pd.to_numeric(items[c], errors="coerce")
    if "product_norm" not in items.columns:
        if "product_raw" in items.columns:
            items["product_norm"] = items["product_raw"].fillna("").str.strip().str.lower()
        else:
            items["product_norm"] = ""

# --------- KPIs ----------
n_receipts = receipts.shape[0]
n_items    = items.shape[0]
if not receipts.empty and "total" in receipts.columns and receipts["total"].notna().any():
    overall_total = float(receipts["total"].fillna(0).sum())
elif not items.empty and "line_total" in items.columns:
    overall_total = float(items["line_total"].fillna(0).sum())
else:
    overall_total = 0.0

# also compute items total for reconciliation section
items_total = float(items["line_total"].fillna(0).sum()) if ("line_total" in items.columns and not items.empty) else 0.0

# --------- time-based (USE RECEIPT TOTALS to avoid double count) ----------
by_month = pd.DataFrame()
by_week  = pd.DataFrame()
by_tod   = pd.DataFrame()  # type of day
by_wd    = pd.DataFrame()  # weekday

if not receipts.empty and receipts["date"].notna().any() and "total" in receipts.columns:
    rc = receipts.dropna(subset=["date"]).copy()

    # by month (YYYY-MM)
    rc["month"] = rc["date"].dt.to_period("M").astype(str)
    by_month = (rc.groupby("month", as_index=False)["total"].sum()
                  .sort_values("month"))

    # by ISO week (YYYY-Www)
    rc["year_week"] = rc["date"].dt.strftime("%G-W%V")
    by_week = (rc.groupby("year_week", as_index=False)["total"].sum()
                 .sort_values("year_week"))

    # by type of day (Weekday vs Weekend)
    rc["type_of_day"] = rc["date"].dt.weekday.map(lambda d: "Weekend" if d >= 5 else "Weekday")
    by_tod = rc.groupby("type_of_day", as_index=False)["total"].sum().sort_values("type_of_day")

    # by weekday name (Mon..Sun)
    rc["weekday"] = rc["date"].dt.day_name()
    order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
    by_wd = (rc.groupby("weekday", as_index=False)["total"].sum()
               .set_index("weekday").reindex(order).fillna(0).reset_index())

# --------- items joined with dates for item-based slices ----------
if not items.empty:
    if "receipt_id" in items.columns and "receipt_id" in receipts.columns:
        df = items.merge(receipts[["receipt_id","date"]], on="receipt_id", how="left")
    else:
        df = items.copy()
        if "date" not in df:
            df["date"] = pd.NaT
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
else:
    df = items.copy()

# --------- most frequent items (with light normalization) ----------
top_items = pd.DataFrame()
if not df.empty and "qty" in df.columns:
    df["freq_name"] = df["product_norm"].map(normalize_for_frequency)
    freq = (df[df["freq_name"].str.len() > 0]
              .groupby("freq_name", as_index=False)["qty"].sum()
              .sort_values("qty", ascending=False))
    top_items = freq.head(25)

# --------- price change (last vs previous per item) ----------
price_change = pd.DataFrame(columns=[
    "item","prev_date","prev_price","last_date","last_price","change_abs","change_pct","days_between"
])

if not df.empty:
    tmp = df.copy()
    # pick a display name
    if "product_norm" not in tmp.columns or tmp["product_norm"].fillna("").eq("").all():
        tmp["item"] = tmp.get("freq_name", tmp.get("product_raw", "")).fillna("").astype(str)
    else:
        tmp["item"] = tmp["product_norm"].fillna("").astype(str)

    # derive unit_price if missing
    if "unit_price" in tmp.columns:
        need = tmp["unit_price"].isna()
        tmp.loc[need, "unit_price"] = (tmp["line_total"] / tmp["qty"]).where(tmp["qty"] > 0)
    else:
        tmp["unit_price"] = (tmp["line_total"] / tmp["qty"]).where(tmp["qty"] > 0)

    tmp = tmp.dropna(subset=["unit_price"])
    tmp["date"] = pd.to_datetime(tmp["date"], errors="coerce")
    tmp = tmp[tmp["date"].notna()].sort_values(["item","date"])

    last2 = tmp.groupby("item").tail(2)

    def summarize(g: pd.DataFrame):
        if len(g) < 2:
            return None
        g = g.sort_values("date")
        prev, last = g.iloc[-2], g.iloc[-1]
        pp, lp = float(prev["unit_price"]), float(last["unit_price"])
        change_abs = lp - pp
        change_pct = (change_abs / pp * 100.0) if pp else None
        days_between = (last["date"] - prev["date"]).days
        return pd.Series({
            "item": g.name,
            "prev_date": prev["date"].date().isoformat(),
            "prev_price": round(pp,2),
            "last_date": last["date"].date().isoformat(),
            "last_price": round(lp,2),
            "change_abs": round(change_abs,2),
            "change_pct": round(change_pct,2) if change_pct is not None else None,
            "days_between": days_between
        })

    pc = last2.groupby("item").apply(summarize).dropna().reset_index(drop=True)
    price_change = pc.sort_values(["change_pct","change_abs"], ascending=[False, False]).head(50)

# --------- category breakdown (learned mapping only; no hardcoding) ----------
category_breakdown = pd.DataFrame(columns=["category","spend"])
if os.path.exists(LEARNED_CATS) and os.path.getsize(LEARNED_CATS) > 0 and not df.empty:
    catmap = pd.read_csv(LEARNED_CATS)
    if "product_norm" in catmap.columns and "category_name" in catmap.columns:
        tmp = df.merge(catmap[["product_norm","category_name"]]
                       .rename(columns={"category_name":"category"}), on="product_norm", how="left")
        tmp["category"] = tmp["category"].fillna("other")
        category_breakdown = (tmp.groupby("category", as_index=False)["line_total"]
                                .sum().rename(columns={"line_total":"spend"})
                                .sort_values("spend", ascending=False))

# --------- build HTML ----------
STYLE = """
<style>
 body{font-family:-apple-system,BlinkMacSystemFont,Segoe UI,Roboto,Helvetica,Arial,sans-serif;margin:24px;}
 h1{margin:0 0 12px 0}
 h3{margin:22px 0 8px}
 .kpis{display:flex;gap:16px;flex-wrap:wrap;margin:12px 0 8px}
 .kpi{padding:12px 16px;border:1px solid #eee;border-radius:12px;box-shadow:0 1px 2px rgba(0,0,0,.04)}
 .kpi .label{font-size:12px;color:#666}
 .kpi .value{font-size:22px;font-weight:600}
 .tbl{border-collapse:collapse;width:100%;}
 .tbl th,.tbl td{padding:8px 10px;border-bottom:1px solid #eee;font-size:14px;text-align:left}
 em{color:#666}
</style>
"""

parts = []

# KPIs
parts.append(f"""
<h1>Naturalia — Spend Dashboard</h1>
<div class="kpis">
  <div class="kpi"><div class="label">Receipts</div><div class="value">{n_receipts}</div></div>
  <div class="kpi"><div class="label">Line items</div><div class="value">{n_items}</div></div>
  <div class="kpi"><div class="label">Total Spend</div><div class="value">{fmt_eur(overall_total)}</div></div>
</div>
""")

# Reconciliation (receipts vs items)
parts.append(f"""
<h3>Reconciliation (Items vs Receipts)</h3>
<table class="tbl">
  <tr><th>Source</th><th>Amount</th></tr>
  <tr><td>Receipts total</td><td>{fmt_eur(overall_total)}</td></tr>
  <tr><td>Items total</td><td>{fmt_eur(items_total)}</td></tr>
  <tr><td>Delta</td><td>{fmt_eur(items_total - overall_total)}</td></tr>
</table>
""")

# Time slices (receipt-based)
parts.append("<h3>Spend by Month</h3>")
parts.append(table(by_month, ["month","total"], ["Month","Spend"], "No dated receipts"))

parts.append("<h3>Spend by ISO Week</h3>")
parts.append(table(by_week, ["year_week","total"], ["Week","Spend"], "No dated receipts"))

parts.append("<h3>Spend by Type of Day</h3>")
parts.append(table(by_tod, ["type_of_day","total"], ["Type of Day","Spend"], "No dated receipts"))

parts.append("<h3>Spend by Weekday</h3>")
parts.append(table(by_wd, ["weekday","total"], ["Weekday","Spend"], "No dated receipts"))

# Items-based insights
parts.append("<h3>Most Frequently Bought Items (Top 25 by Qty)</h3>")
parts.append(table(top_items, ["freq_name","qty"], ["Item","Total Qty"], "No repeated items yet"))

parts.append("<h3>Price Change (Last vs Previous)</h3>")
parts.append(table(
    price_change,
    ["item","prev_date","prev_price","last_date","last_price","change_abs","change_pct","days_between"],
    ["Item","Prev Date","Prev Price","Last Date","Last Price","Δ Price","Δ %","Days Between"],
    "Not enough observations yet"
))

# Categories (learned)
parts.append("<h3>Spend by Category (Learned)</h3>")
parts.append(table(category_breakdown, ["category","spend"], ["Category","Spend"], "No learned categories yet"))

html = (
    "<!doctype html><html><head><meta charset='utf-8'>"
    "<title>Naturalia — Spend Dashboard</title>"
    f"{STYLE}</head><body>" + "".join(parts) + "</body></html>"
)

with open(OUT_HTML, "w", encoding="utf-8") as f:
    f.write(html)

print("Wrote", OUT_HTML)
