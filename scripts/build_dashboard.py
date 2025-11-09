# scripts/build_dashboard.py
import os, re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

RCPT = "data/receipts.csv"
ITEM = "data/items.csv"
OUT_DIR = "artifacts"
OUT_HTML = os.path.join(OUT_DIR, "dashboard.html")
os.makedirs(OUT_DIR, exist_ok=True)

print("== Naturalia spend analysis ==")

# ------------------ Load & normalize ------------------
rc = pd.read_csv(RCPT, low_memory=False)
it = pd.read_csv(ITEM, low_memory=False)

rc["date"]  = pd.to_datetime(rc.get("date"), errors="coerce")
rc["total"] = pd.to_numeric(rc.get("total"), errors="coerce").fillna(0)

it["qty"]        = pd.to_numeric(it.get("qty"), errors="coerce").fillna(1)
it["unit_price"] = pd.to_numeric(it.get("unit_price"), errors="coerce")
it["line_total"] = pd.to_numeric(it.get("line_total"), errors="coerce")

print(f"Receipts: {len(rc)}, Items: {len(it)}")

# ------------------ Helpers ------------------
def fmt_eur(x):
    try:
        v = float(x) if pd.notna(x) else 0.0
        return f"€{v:,.2f}".replace(",", " ")
    except Exception:
        return "€0.00"

def table_html(df: pd.DataFrame, cols, header=None, empty="No data"):
    if df is None or df.empty:
        return f"<em>{empty}</em>"
    use = [c for c in cols if c in df.columns]
    t = df.loc[:, use].copy()
    for c in t.columns:
        if any(k in c for k in ("total", "price", "spend", "value")):
            t[c] = t[c].apply(fmt_eur)
    if header:
        t.columns = header
    return t.to_html(index=False, border=0, classes="tbl")

# ======================================================
# A) Time-slice totals
# ======================================================
rc_valid = rc.dropna(subset=["date"]).copy()
grand_total = float(rc_valid["total"].sum())

by_month = (rc_valid
            .assign(month=rc_valid["date"].dt.to_period("M").astype(str))
            .groupby("month", as_index=False)["total"].sum()
            .sort_values("month"))

by_week = (rc_valid
           .assign(week=rc_valid["date"].dt.strftime("%G-W%V"))
           .groupby("week", as_index=False)["total"].sum()
           .sort_values("week"))

by_wday = (rc_valid
           .assign(weekday=rc_valid["date"].dt.day_name())
           .groupby("weekday", as_index=False)["total"].sum())

print("\n=== A) Totals by time slices ===")
print(f"Grand total: {grand_total:.2f}")
print(f"Sum by month: {float(by_month['total'].sum()):.2f}")
print(f"Sum by week : {float(by_week ['total'].sum()):.2f}")
print(f"Sum by wday : {float(by_wday ['total'].sum()):.2f}")
print("\n-- by month --");  print(by_month.to_string(index=False))
print("\n-- by week --");   print(by_week.to_string(index=False))
print("\n-- by weekday --");print(by_wday.to_string(index=False))

# ======================================================
# B) Price evolution across receipts (warning-free)
# ======================================================
print("\n=== B) Price evolution across receipts (warning-free) ===")

name_col = ("product" if "product" in it.columns else
            ("product_norm" if "product_norm" in it.columns else
             ("product_raw" if "product_raw" in it.columns else None)))
if name_col is None:
    name_col = "product"
    it[name_col] = ""

# choose a join key present on both sides
join_key = None
for cand in ("receipt_uid", "receipt_id"):
    if cand in it.columns and cand in rc_valid.columns:
        join_key = cand; break
if join_key is None: join_key = "receipt_id"

tmp = it.copy()
tmp["unit_price"] = pd.to_numeric(tmp.get("unit_price"), errors="coerce")
tmp["unit_price"] = tmp["unit_price"].fillna(
    pd.to_numeric(tmp.get("line_total"), errors="coerce") / tmp["qty"]
)

rc_join = rc_valid[[join_key, "date"]].rename(columns={join_key: "join_key"})
tmp["join_key"] = tmp[join_key] if join_key in tmp.columns else np.nan
tmp = tmp.merge(rc_join, on="join_key", how="left").drop(columns=["join_key"])
tmp["date"] = pd.to_datetime(tmp["date"], errors="coerce")

# keep items that appear on >=2 different receipts
if join_key in tmp.columns:
    counts = tmp.groupby(name_col)[join_key].nunique()
    repeated_names = counts[counts >= 2].index
    tmp = tmp[tmp[name_col].isin(repeated_names)]
else:
    tmp = tmp.iloc[0:0]

if tmp.empty:
    print("No repeated product names across different receipts yet.")
    price_change = pd.DataFrame(
        columns=["item","prev_date","prev_price","last_date","last_price","Δ_price","Δ_%","days_between"]
    )
else:
    s = tmp.sort_values([name_col, "date"])
    idx_last = s.groupby(name_col)["date"].idxmax()
    last = s.loc[idx_last, [name_col, "date", "unit_price"]].rename(
        columns={"date": "last_date", "unit_price": "last_price"}
    )
    s_wo_last = s.drop(index=idx_last)
    idx_prev = s_wo_last.groupby(name_col)["date"].idxmax()
    prev = s_wo_last.loc[idx_prev, [name_col, "date", "unit_price"]].rename(
        columns={"date": "prev_date", "unit_price": "prev_price"}
    )
    pc = prev.merge(last, on=name_col, how="inner")
    pc["Δ_price"] = (pc["last_price"] - pc["prev_price"]).round(2)
    pc["Δ_%"] = ((pc["last_price"] - pc["prev_price"]) / pc["prev_price"] * 100).round(2)
    pc["days_between"] = (
        pd.to_datetime(pc["last_date"]) - pd.to_datetime(pc["prev_date"])
    ).dt.days
    price_change = pc.rename(columns={name_col: "item"})[
        ["item","prev_date","prev_price","last_date","last_price","Δ_price","Δ_%","days_between"]
    ].sort_values(["Δ_%","Δ_price"], ascending=[False, False]).reset_index(drop=True)

if not price_change.empty:
    print(price_change.to_string(index=False))

# ======================================================
# C) Machine-learned categories (French-aware)
# ======================================================
print("\n=== C) Machine-learned categories (French receipts) ===")

def clean_fr(s: str) -> str:
    s = (s or "").lower()
    s = re.sub(r"\d+[x×]?\s*\d*(ml|l|g|kg|cl|pack|pcs)?", " ", s)
    s = re.sub(r"[^a-zàâäéèêëîïôöùûüç\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

it["clean"] = it[name_col].fillna("").map(clean_fr)
it["line_total"] = pd.to_numeric(it["line_total"], errors="coerce").fillna(0)

stop_fr = {
    "de","du","des","le","la","les","un","une","et","ou","en","au","aux",
    "pour","avec","sans","sur","sous","dans","à","par","chez","tout","tous"
}

vect = TfidfVectorizer(min_df=1, ngram_range=(1,2), stop_words=list(stop_fr))
X = vect.fit_transform(it["clean"])

k = min(max(5, len(it)//5), 15)
km = KMeans(n_clusters=k, n_init="auto", random_state=42)
labels = km.fit_predict(X)
it["category_id"] = labels

terms = vect.get_feature_names_out()
def top_terms(center, n=3):
    idx = center.argsort()[-n:][::-1]
    return "|".join(terms[i] for i in idx)

cat_names = [top_terms(c) for c in km.cluster_centers_]
it["category_name"] = [cat_names[i] for i in labels]

cat = (it.groupby("category_name", as_index=False)["line_total"]
         .sum().rename(columns={"line_total":"spend"})
         .sort_values("spend", ascending=False))

print(f"Clusters learned (k={k})")
print("-- Top categories by spend --")
print(cat.to_string(index=False))
print(f"\nSum of category spend: {float(cat['spend'].sum()):.2f}")
print(f"Sum of all item totals : {float(it['line_total'].sum()):.2f}")

print("\n-- Example items per cluster --")
for name, sub in it.groupby("category_name"):
    ex = ", ".join(sub[name_col].head(5))
    print(f"{name}: {ex}")

# persist learned mapping (product → category_name)
it[[name_col,"category_name"]].drop_duplicates().rename(columns={name_col:"product"}) \
  .to_csv(os.path.join(OUT_DIR, "categories_learned.csv"), index=False)

# ======================================================
# D) Item price percentiles (unit_price)
# ======================================================
print("\n=== D) Item price distribution (unit price percentiles) ===")
it["unit_price"] = it["unit_price"].fillna(it["line_total"]/it["qty"])
prices = it["unit_price"].dropna().astype(float).values

if prices.size == 0:
    pct_df = pd.DataFrame({"percentile": [], "value": []})
else:
    percentiles = [10,25,50,75,90]
    p_values = np.percentile(prices, percentiles)
    for p, val in zip(percentiles, p_values):
        print(f"P{p:02}: {val:.2f} €")
    print(f"Min: {prices.min():.2f} €,  Max: {prices.max():.2f} €,  Mean: {prices.mean():.2f} €")
    pct_df = pd.DataFrame({"percentile": [f"P{p}" for p in percentiles], "value": p_values})
pct_df.to_csv(os.path.join(OUT_DIR,"price_percentiles.csv"), index=False)

# ======================================================
# HTML build (no charts)
# ======================================================
STYLE = """
<style>
 body{font-family:-apple-system,BlinkMacSystemFont,Segoe UI,Roboto,Helvetica,Arial,sans-serif;margin:24px;}
 h1{margin:0 0 12px 0} h3{margin:22px 0 8px}
 .kpis{display:flex;gap:16px;flex-wrap:wrap;margin:12px 0 8px}
 .kpi{padding:12px 16px;border:1px solid #eee;border-radius:12px;box-shadow:0 1px 2px rgba(0,0,0,.04)}
 .kpi .label{font-size:12px;color:#666} .kpi .value{font-size:22px;font-weight:600}
 .tbl{border-collapse:collapse;width:100%;}
 .tbl th,.tbl td{padding:8px 10px;border-bottom:1px solid #eee;font-size:14px;text-align:left}
 em{color:#666}
</style>
"""

items_total = float(it["line_total"].fillna(0).sum()) if "line_total" in it.columns else 0.0

parts = []
# KPIs + reconciliation
parts.append(f"""
<h1>Naturalia — Spend Dashboard</h1>
<div class="kpis">
  <div class="kpi"><div class="label">Receipts</div><div class="value">{len(rc)}</div></div>
  <div class="kpi"><div class="label">Items</div><div class="value">{len(it)}</div></div>
  <div class="kpi"><div class="label">Receipts Total</div><div class="value">{fmt_eur(grand_total)}</div></div>
</div>
<h3>Reconciliation</h3>
<table class="tbl">
  <tr><th>Source</th><th>Amount</th></tr>
  <tr><td>Receipts total</td><td>{fmt_eur(grand_total)}</td></tr>
  <tr><td>Items total</td><td>{fmt_eur(items_total)}</td></tr>
  <tr><td>Delta</td><td>{fmt_eur(items_total - grand_total)}</td></tr>
</table>
""")

# time slices
parts.append("<h3>Spend by Month</h3>")
parts.append(table_html(by_month, ["month","total"], ["Month","Spend"], "No dated receipts"))

parts.append("<h3>Spend by ISO Week</h3>")
parts.append(table_html(by_week, ["week","total"], ["Week","Spend"], "No dated receipts"))

parts.append("<h3>Spend by Weekday</h3>")
parts.append(table_html(by_wday, ["weekday","total"], ["Weekday","Spend"], "No dated receipts"))

# price change
parts.append("<h3>Price Change (Last vs Previous)</h3>")
parts.append(table_html(
    price_change,
    ["item","prev_date","prev_price","last_date","last_price","Δ_price","Δ_%","days_between"],
    ["Item","Prev Date","Prev Price","Last Date","Last Price","Δ Price","Δ %","Days Between"],
    "No repeated products yet"
))

# categories
parts.append("<h3>Spend by Category (Learned)</h3>")
parts.append(table_html(cat, ["category_name","spend"], ["Category","Spend"], "No learned categories yet"))

# percentiles
parts.append("<h3>Price Percentiles (Unit Price)</h3>")
parts.append(table_html(pct_df, ["percentile","value"], ["Percentile","Value"], "No prices"))

html = "<!doctype html><html><head><meta charset='utf-8'><title>Naturalia — Dashboard</title>" + STYLE + "</head><body>" + "".join(parts) + "</body></html>"
with open(OUT_HTML, "w", encoding="utf-8") as f:
    f.write(html)

print(f"\nWrote {OUT_HTML}")
