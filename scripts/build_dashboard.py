# scripts/build_dashboard.py
import os, re
import pandas as pd

RCPT = "data/receipts.csv"
ITEM = "data/items.csv"
OUT_DIR = "artifacts"
OUT_HTML = os.path.join(OUT_DIR, "dashboard.html")
LEARNED_CATS = "artifacts/categories_learned.csv"  # optional

os.makedirs(OUT_DIR, exist_ok=True)

# ---------- helpers ----------
def read_csv_safe(path, **kw):
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_csv(path, **kw)

def fmt_eur(x):
    try:
        v = float(x) if pd.notna(x) else 0.0
        return f"€{v:,.2f}".replace(",", " ")
    except Exception:
        return "€0.00"

def table(df, cols, header=None, empty_msg="No data"):
    if df is None or df.empty:
        return f"<em>{empty_msg}</em>"
    use = [c for c in cols if c in df.columns]
    if not use:
        return f"<em>{empty_msg}</em>"
    t = df.loc[:, use].copy()
    for c in t.columns:
        if any(k in c for k in ("total", "price", "spend")) or c in ("items_sum",):
            t[c] = t[c].apply(fmt_eur)
    if header:
        t.columns = header
    return t.to_html(index=False, border=0, classes="tbl")

def norm_freq_name(s: str) -> str:
    s = (s or "").lower()
    s = re.sub(r"\b\d+\s*(x|×)?\s*\d*\s*(ml|l|g|kg|cl|pack|pcs)\b", " ", s)
    s = re.sub(r"\b\d+\s*(ml|l|g|kg|cl)\b", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

# ---------- load ----------
receipts = read_csv_safe(RCPT)
items    = read_csv_safe(ITEM)

# receipts: dates & nums
if not receipts.empty and "date" in receipts.columns:
    receipts["date"] = pd.to_datetime(receipts["date"], errors="coerce")
else:
    receipts["date"] = pd.NaT
for c in ("total","subtotal","tax","confidence"):
    if c in receipts.columns:
        receipts[c] = pd.to_numeric(receipts[c], errors="coerce")

# items: nums & names
if not items.empty:
    for c in ("qty","unit_price","line_total"):
        if c in items.columns:
            items[c] = pd.to_numeric(items[c], errors="coerce")
    if "product_norm" not in items.columns:
        items["product_norm"] = items.get("product_raw","").fillna("").astype(str).str.lower()

# ---------- KPIs ----------
n_receipts = receipts.shape[0]
n_items    = items.shape[0]
receipts_total = float(receipts["total"].fillna(0).sum()) if ("total" in receipts.columns) else 0.0
items_total    = float(items["line_total"].fillna(0).sum()) if ("line_total" in items.columns) else 0.0
overall_total  = receipts_total or items_total

# ---------- receipt-based time slices (avoid double count) ----------
by_month = pd.DataFrame(); by_week = pd.DataFrame(); by_tod = pd.DataFrame(); by_wd = pd.DataFrame()
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
               .set_index("weekday").reindex(order).fillna(0).reset_index())

# ---------- items joined to dates ----------
if not items.empty:
    if {"receipt_id"}.issubset(items.columns) and {"receipt_id"}.issubset(receipts.columns):
        df = items.merge(receipts[["receipt_id","date"]], on="receipt_id", how="left")
    else:
        df = items.copy(); df["date"] = pd.NaT
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
else:
    df = items.copy()

# ---------- most frequent items (name + qty) ----------
top_items = pd.DataFrame()
if not df.empty and "qty" in df.columns:
    # pick a display name with fallback so it never collapses to a single number
    disp = df["product_norm"].fillna("")
    fallback = df.get("product_raw","").fillna("")
    disp = disp.where(disp.str.len() > 0, fallback.astype(str).str.lower())
    df["freq_name"] = disp.map(norm_freq_name)
    freq = (df[df["freq_name"].str.len() > 0]
              .groupby("freq_name", as_index=False)["qty"].sum()
              .sort_values("qty", ascending=False))
    top_items = freq.head(25)

# ---------- price change (clear item & dates) ----------
price_change = pd.DataFrame(columns=[
    "item","prev_date","prev_price","last_date","last_price","change_abs","change_pct","days_between"
])
if not df.empty:
    tmp = df.copy()
    # display item name
    disp = tmp["product_norm"].fillna("")
    fallback = tmp.get("freq_name", tmp.get("product_raw","")).fillna("")
    tmp["item"] = disp.where(disp.str.len() > 0, fallback.astype(str).str.lower())

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

    def _summ(g: pd.DataFrame):
        if len(g) < 2: return None
        g = g.sort_values("date")
        prev, last = g.iloc[-2], g.iloc[-1]
        pp, lp = float(prev["unit_price"]), float(last["unit_price"])
        ch_abs = lp - pp
        ch_pct = (ch_abs / pp * 100.0) if pp else None
        days   = (last["date"] - prev["date"]).days
        return pd.Series({
            "item": g.name,
            "prev_date": prev["date"].date().isoformat(),
            "prev_price": round(pp,2),
            "last_date": last["date"].date().isoformat(),
            "last_price": round(lp,2),
            "change_abs": round(ch_abs,2),
            "change_pct": round(ch_pct,2) if ch_pct is not None else None,
            "days_between": days
        })

    pc = last2.groupby("item").apply(_summ).dropna().reset_index(drop=True)
    price_change = pc.sort_values(["change_pct","change_abs"], ascending=[False, False]).head(50)

# ---------- categories (learned only; show when available) ----------
category_breakdown = pd.DataFrame(columns=["category","spend"])
if os.path.exists(LEARNED_CATS) and os.path.getsize(LEARNED_CATS) > 0 and not df.empty:
    catmap = pd.read_csv(LEARNED_CATS)
    if {"product_norm","category_name"}.issubset(catmap.columns):
        tmp = df.merge(catmap[["product_norm","category_name"]]
                       .rename(columns={"category_name":"category"}), on="product_norm", how="left")
        tmp["category"] = tmp["category"].fillna("other")
        category_breakdown = (tmp.groupby("category", as_index=False)["line_total"]
                                .sum().rename(columns={"line_total":"spend"})
                                .sort_values("spend", ascending=False))

# ---------- HTML ----------
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
parts.append(f"""
<h1>Naturalia — Spend Dashboard</h1>
<div class="kpis">
  <div class="kpi"><div class="label">Receipts</div><div class="value">{n_receipts}</div></div>
  <div class="kpi"><div class="label">Line items</div><div class="value">{n_items}</div></div>
  <div class="kpi"><div class="label">Total Spend</div><div class="value">{fmt_eur(overall_total)}</div></div>
</div>
""")

# reconciliation
parts.append(f"""
<h3>Reconciliation (Items vs Receipts)</h3>
<table class="tbl">
  <tr><th>Source</th><th>Amount</th></tr>
  <tr><td>Receipts total</td><td>{fmt_eur(receipts_total)}</td></tr>
  <tr><td>Items total</td><td>{fmt_eur(items_total)}</td></tr>
  <tr><td>Delta</td><td>{fmt_eur(items_total - receipts_total)}</td></tr>
</table>
""")

# time slices
parts.append("<h3>Spend by Month</h3>")
parts.append(table(by_month, ["month","total"], ["Month","Spend"], "No dated receipts"))

parts.append("<h3>Spend by ISO Week</h3>")
parts.append(table(by_week, ["year_week","total"], ["Week","Spend"], "No dated receipts"))

parts.append("<h3>Spend by Type of Day</h3>")
parts.append(table(by_tod, ["type_of_day","total"], ["Type of Day","Spend"], "No dated receipts"))

parts.append("<h3>Spend by Weekday</h3>")
parts.append(table(by_wd, ["weekday","total"], ["Weekday","Spend"], "No dated receipts"))

# items insights
parts.append("<h3>Most Frequently Bought Items (Top 25 by Qty)</h3>")
parts.append(table(top_items, ["freq_name","qty"], ["Item","Total Qty"], "No repeated items yet"))

parts.append("<h3>Price Change (Last vs Previous)</h3>")
parts.append(table(
    price_change,
    ["item","prev_date","prev_price","last_date","last_price","change_abs","change_pct","days_between"],
    ["Item","Prev Date","Prev Price","Last Date","Last Price","Δ Price","Δ %","Days Between"],
    "Not enough observations yet"
))

# categories
parts.append("<h3>Spend by Category (Learned)</h3>")
parts.append(table(category_breakdown, ["category","spend"], ["Category","Spend"], "No learned categories yet"))

html = "<!doctype html><html><head><meta charset='utf-8'><title>Naturalia — Spend Dashboard</title>" + STYLE + "</head><body>" + "".join(parts) + "</body></html>"

with open(OUT_HTML, "w", encoding="utf-8") as f:
    f.write(html)

print("Wrote", OUT_HTML)
