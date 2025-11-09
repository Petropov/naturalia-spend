# scripts/build_dashboard.py
import os, re
import pandas as pd

RCPT = "data/receipts.csv"
ITEM = "data/items.csv"
OUT_DIR = "artifacts"
OUT_HTML = os.path.join(OUT_DIR, "dashboard.html")
LEARNED_CATS = "artifacts/categories_learned.csv"  # produced by scripts/learn_categories.py

os.makedirs(OUT_DIR, exist_ok=True)

def read_csv_safe(path, **kw):
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_csv(path, **kw)

# ---------- Load & normalize ----------
receipts = read_csv_safe(RCPT)
items    = read_csv_safe(ITEM)

# Dates / numbers on receipts
if not receipts.empty and "date" in receipts.columns:
    receipts["date"] = pd.to_datetime(receipts["date"], errors="coerce")
else:
    receipts["date"] = pd.NaT
for c in ("total","subtotal","tax","confidence"):
    if c in receipts.columns:
        receipts[c] = pd.to_numeric(receipts[c], errors="coerce")

# Numbers on items
if not items.empty:
    for c in ("qty","unit_price","line_total"):
        if c in items.columns:
            items[c] = pd.to_numeric(items[c], errors="coerce")
    # Ensure product_norm exists
    if "product_norm" not in items.columns:
        if "product_raw" in items.columns:
            items["product_norm"] = items["product_raw"].fillna("").str.strip().str.lower()
        else:
            items["product_norm"] = ""

# ---------- KPIs ----------
n_receipts = receipts.shape[0]
n_items    = items.shape[0]
if not receipts.empty and "total" in receipts.columns and receipts["total"].notna().any():
    overall_total = float(receipts["total"].fillna(0).sum())
elif not items.empty and "line_total" in items.columns:
    overall_total = float(items["line_total"].fillna(0).sum())
else:
    overall_total = 0.0

# ---------- Time-based (use RECEIPT totals to avoid double count) ----------
by_month = pd.DataFrame()
by_week  = pd.DataFrame()
by_tod   = pd.DataFrame()   # type of day
by_wd    = pd.DataFrame()   # weekday

if not receipts.empty and receipts["date"].notna().any() and "total" in receipts.columns:
    rc = receipts.dropna(subset=["date"]).copy()

    rc["month"] = rc["date"].dt.to_period("M").astype(str)
    by_month = rc.groupby("month", as_index=False)["total"].sum().sort_values("month")

    rc["year_week"] = rc["date"].dt.strftime("%G-W%V")
    by_week = rc.groupby("year_week", as_index=False)["total"].sum().sort_values("year_week")

    rc["type_of_day"] = rc["date"].dt.weekday.map(lambda d: "Weekend" if d >= 5 else "Weekday")
    by_tod = rc.groupby("type_of_day", as_index=False)["total"].sum().sort_values("type_of_day")

    rc["weekday"] = rc["date"].dt.day_name()
    order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
    by_wd = (rc.groupby("weekday", as_index=False)["total"].sum()
               .set_index("weekday").reindex(order).reset_index())

# ---------- Join items to dates for item-based slices ----------
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

# ---------- Most frequent items (with light normalization) ----------
def normalize_for_frequency(s: str) -> str:
    s = (s or "").lower()
    # strip pack sizes like "4x125g", "500 ml", "2×", etc.
    s = re.sub(r"\b\d+\s*(x|×)?\s*\d*\s*(ml|l|g|kg|cl|pack|pcs)\b", " ", s)
    # strip trailing sizes like "500g" anywhere
    s = re.sub(r"\b\d+\s*(ml|l|g|kg|cl)\b", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

top_items = pd.DataFrame()
if not df.empty and "qty" in df.columns:
    df["freq_name"] = df["product_norm"].map(normalize_for_frequency)
    grp = (df.groupby("freq_name", as_index=False)["qty"].sum()
             .sort_values("qty", ascending=False))
    top_items = grp.head(25)

# ---------- Price change (last vs previous per item) ----------
price_change = pd.DataFrame(columns=[
    "product_norm","prev_date","prev_price","last_date","last_price","change_abs","change_pct","days_between"
])
if not df.empty and "product_norm" in df.columns:
    tmp = df.copy()
    # derive unit_price if missing
    if "unit_price" in tmp.columns:
        need = tmp["unit_price"].isna()
        tmp.loc[need, "unit_price"] = (tmp["line_total"] / tmp["qty"]).where(tmp["qty"] > 0)
    else:
        tmp["unit_price"] = (tmp["line_total"] / tmp["qty"]).where(tmp["qty"] > 0)
    tmp = tmp.dropna(subset=["unit_price"])
    tmp["date"] = pd.to_datetime(tmp["date"], errors="coerce")
    tmp = tmp.sort_values(["product_norm","date"])

    last2 = tmp.groupby("product_norm").tail(2)
    def summarize(g: pd.DataFrame):
        if len(g) < 2: return None
        g = g.sort_values("date")
        prev, last = g.iloc[-2], g.iloc[-1]
        if pd.isna(prev["unit_price"]) or pd.isna(last["unit_price"]):
            return None
        pp, lp = float(prev["unit_price"]), float(last["unit_price"])
        pdte = prev["date"]; ldte = last["date"]
        change_abs = lp - pp
        change_pct = (change_abs / pp * 100.0) if pp else None
        days_between = (ldte - pdte).days if pd.notna(pdte) and pd.notna(ldte) else None
        return pd.Series({
            "product_norm": g.name,
            "prev_date": pdte.date().isoformat() if pd.notna(pdte) else "",
            "prev_price": round(pp,2),
            "last_date": ldte.date().isoformat() if pd.notna(ldte) else "",
            "last_price": round(lp,2),
            "change_abs": round(change_abs,2),
            "change_pct": round(change_pct,2) if change_pct is not None else None,
            "days_between": days_between
        })
    pc = last2.groupby("product_norm").apply(summarize).dropna().reset_index(drop=True)
    price_change = pc.sort_values(["change_pct","change_abs"], ascending=[False, False]).head(50)

# ---------- Category breakdown (learned mapping only; no hardcoding) ----------
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

# ---------- Helpers ----------
def fmt_eur(x):
    try:
        return f"€{float(x):,.2f}".replace(",", " ").replace("\xa0"," ")
    except Exception:
        return ""

def table(df, cols, header=None, empty_msg="No data"):
    if df is None or df.empty:
        return f"<em>{empty_msg}</em>"
    use = [c for c in cols if c in df.columns]
    t = df.loc[:, use].copy()
    # format money-looking columns
    for c in t.columns:
        if any(k in c for k in ("total","price","spend")) or c in ("items_sum",):
            t[c] = t[c].apply(fmt_eur)
    if header:
        t.columns = header
    return t.to_html(index=False, border=0, classes="tbl")

# ---------- Compose HTML ----------
parts = []
parts.append(f"""
<h1>Naturalia — Spend Dashboard</h1>
<div class="kpis">
  <div class="kpi"><div class="label">Receipts</div><div class="value">{n_receipts}</div></div>
  <div class="kpi"><div class="label">Line items</div><div class="value">{n_items}</div></div>
  <div class="kpi"><div class="label">Total Spend</div><div class="value">{fmt_eur(overall_total)}</div></div>
</div>
""")

parts.append("<h3>Spend by Month</h3>")
parts.append(table(by_month, ["month","total"], ["Month","Spend"], "No dated receipts"))

parts.append("<h3>Spend by ISO Week</h3>")
parts.append(table(by_week, ["year_week","total"], ["Week","Spend"], "No dated receipts"))

parts.append("<h3>Spend by Type of Day</h3>")
parts.append(table(by_tod, ["type_of_day","total"], ["Type of Day","Spend"], "No dated receipts"))

parts.append("<h3>Spend by Weekday</h3>")
parts.append(table(by_wd, ["weekday","total"], ["Weekday","Spend"], "No dated receipts"))

parts.append("<h3>Most Frequently Bought Items (Top 25 by Qty)</h3>")
parts.append(table(top_items, ["freq_name","qty"], ["Item","Total Qty"], "No repeated items yet"))

parts.append("<h3>Price Change (Last vs Previous)</h3>")
parts.append(table(
    price_change,
    ["product_norm","prev_date","prev_price","last_date","last_price","change_abs","change_pct","days_between"],
    ["Item","Prev Date","Prev Price","Last Date","Last Price","Δ Price","Δ %","Days Between"],
    "Not enough observations yet"
))

parts.append("<h3>Spend by Category (Learned)</h3>")
parts.append(table(category_breakdown, ["category","spend"], ["Category","Spend"], "No learned categories yet"))

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

html = "<!doctype html><html><head><meta charset='utf-8'><title>Naturalia — Spend Dashboard</title>" + STYLE + "</head><body>" + "".join(parts) + "</body></html>"

with open(OUT_HTML, "w", encoding="utf-8") as f:
    f.write(html)

print("Wrote", OUT_HTML)
