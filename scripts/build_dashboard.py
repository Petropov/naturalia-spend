# scripts/build_dashboard.py
import os
import re
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

# ------------------ config / io ------------------
VERSION_TAG = "Naturalia Dashboard v3 — receipts-only A, warning-free B, ML-C (fr), P10..P90, HTML"
RCPT = "data/receipts.csv"
ITEM = "data/items.csv"
OUT_DIR = "artifacts"
OUT_HTML = os.path.join(OUT_DIR, "dashboard.html")
Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

print("== Naturalia spend analysis ==")

# ------------------ load & normalize ------------------
rc = pd.read_csv(RCPT, low_memory=False)
it = pd.read_csv(ITEM, low_memory=False)

# receipts
rc["date"] = pd.to_datetime(rc.get("date"), errors="coerce")
rc["total"] = pd.to_numeric(rc.get("total"), errors="coerce").fillna(0)

# items
it["qty"] = pd.to_numeric(it.get("qty"), errors="coerce").fillna(1)
it["unit_price"] = pd.to_numeric(it.get("unit_price"), errors="coerce")
it["line_total"] = pd.to_numeric(it.get("line_total"), errors="coerce")

print(f"Receipts: {len(rc)}, Items: {len(it)}")

# ======================================================
# A) Time-slice totals (RECEIPTS ONLY, must match grand total)
# ======================================================
rc_valid = rc.dropna(subset=["date"]).copy()
grand_total = float(rc_valid["total"].sum())

r_by_month = (
    rc_valid.assign(month=rc_valid["date"].dt.to_period("M").astype(str))
            .groupby("month", as_index=False)["total"].sum()
            .sort_values("month")
)
r_by_week = (
    rc_valid.assign(week=rc_valid["date"].dt.strftime("%G-W%V"))
            .groupby("week", as_index=False)["total"].sum()
            .sort_values("week")
)
r_by_tod = (
    rc_valid.assign(type_of_day=rc_valid["date"].dt.weekday.map(lambda d: "Weekend" if d >= 5 else "Weekday"))
            .groupby("type_of_day", as_index=False)["total"].sum()
)
r_by_wday = (
    rc_valid.assign(weekday=rc_valid["date"].dt.day_name())
            .groupby("weekday", as_index=False)["total"].sum()
)

# hard checks to prevent regressions
for label, df, col in [
    ("month", r_by_month, "total"),
    ("week",  r_by_week,  "total"),
    ("weekday", r_by_wday, "total"),
    ("type_of_day", r_by_tod, "total"),
]:
    s = float(df[col].sum()) if not df.empty else 0.0
    if abs(s - grand_total) > 1e-6:
        print(f"[warn] Sum by {label} ({s:.2f}) != grand total ({grand_total:.2f})")

print("\n=== A) Totals by time slices ===")
print(f"Grand total: {grand_total:.2f}")
print(f"Sum by month: {float(r_by_month['total'].sum()):.2f}")
print(f"Sum by week : {float(r_by_week ['total'].sum()):.2f}")
print(f"Sum by wday : {float(r_by_wday ['total'].sum()):.2f}")
print("\n-- by month --");  print(r_by_month.to_string(index=False))
print("\n-- by week --");   print(r_by_week.to_string(index=False))
print("\n-- by weekday --");print(r_by_wday.to_string(index=False))

# ======================================================
# B) Price evolution across receipts (warning-free)
#    Uses receipt_uid if available; otherwise joins on receipt_id (best effort).
#    If items can't be mapped to distinct receipts/dates, output is empty.
# ======================================================
print("\n=== B) Price evolution across receipts (warning-free) ===")

# item display name column
name_col = (
    "product" if "product" in it.columns else
    ("product_norm" if "product_norm" in it.columns else
     ("product_raw" if "product_raw" in it.columns else None))
)
if name_col is None:
    name_col = "product"
    it[name_col] = ""

# choose a join key present on both sides
join_key = None
for cand in ("receipt_uid", "receipt_id"):
    if cand in it.columns and cand in rc_valid.columns:
        join_key = cand
        break
# build a mapping from items to a single date per receipt "instance"
tmp = it.copy()
tmp["unit_price"] = pd.to_numeric(tmp.get("unit_price"), errors="coerce")
tmp["unit_price"] = tmp["unit_price"].fillna(tmp["line_total"] / tmp["qty"])

if join_key is not None:
    # dedupe receipt side on [join_key] keeping earliest date so we don't explode rows
    rc_map = (
        rc_valid.sort_values("date")
                .drop_duplicates(subset=[join_key], keep="first")
                [[join_key, "date"]]
                .rename(columns={join_key: "jk"})
    )
    tmp["jk"] = tmp[join_key]
    tmp = tmp.merge(rc_map, on="jk", how="left").drop(columns=["jk"])
else:
    # final fallback: we have no stable key; we can't map items to unique receipts
    tmp["date"] = pd.NaT

tmp["date"] = pd.to_datetime(tmp["date"], errors="coerce")

# keep items that appear on >=2 distinct dates (true repeats)
if tmp["date"].notna().any():
    counts = tmp.groupby(name_col)["date"].nunique()
    repeated = counts[counts >= 2].index
    tmp = tmp[tmp[name_col].isin(repeated)]
else:
    tmp = tmp.iloc[0:0]

if tmp.empty:
    print("No repeated product names across distinct receipts/dates yet.")
    price_change = pd.DataFrame(
        columns=["item","prev_date","prev_price","last_date","last_price","Δ_price","Δ_%","days_between"]
    )
else:
    s = tmp.sort_values([name_col, "date"])
    idx_last = s.groupby(name_col)["date"].idxmax()
    last = (
        s.loc[idx_last, [name_col, "date", "unit_price"]]
         .rename(columns={"date": "last_date", "unit_price": "last_price"})
    )
    s_wo_last = s.drop(index=idx_last)
    idx_prev = s_wo_last.groupby(name_col)["date"].idxmax()
    prev = (
        s_wo_last.loc[idx_prev, [name_col, "date", "unit_price"]]
                 .rename(columns={"date": "prev_date", "unit_price": "prev_price"})
    )
    pc = prev.merge(last, on=name_col, how="inner")
    pc["Δ_price"] = (pc["last_price"] - pc["prev_price"]).round(2)
    pc["Δ_%"] = ((pc["last_price"] - pc["prev_price"]) / pc["prev_price"] * 100).round(2)
    pc["days_between"] = (
        pd.to_datetime(pc["last_date"]) - pd.to_datetime(pc["prev_date"])
    ).dt.days
    price_change = (
        pc.rename(columns={name_col: "item"})
          [["item","prev_date","prev_price","last_date","last_price","Δ_price","Δ_%","days_between"]]
          .sort_values(["Δ_%","Δ_price"], ascending=[False, False])
          .reset_index(drop=True)
    )

if not price_change.empty:
    print(price_change.to_string(index=False))
 
# ======================================================
# C) Machine-learned categories (French-aware) + spend by category
# ======================================================
from scripts.learn_categories import main as learn_cats

learn_cats()  # writes artifacts/*category* files

# Merge learned categories back into items (it) so we can export category_name per product
if "category_name" not in it.columns:
    learned_path = os.path.join(OUT_DIR, "categories_learned.csv")
    if os.path.exists(learned_path):
        learned = pd.read_csv(learned_path)

        # learned file should have: product, category_name
        if "product" in learned.columns and "category_name" in learned.columns:
            it = it.merge(
                learned[["product", "category_name"]].drop_duplicates(),
                left_on=name_col,
                right_on="product",
                how="left",
            ).drop(columns=["product"], errors="ignore")
        else:
            print(f"WARNING: {learned_path} missing 'product'/'category_name' columns; skipping merge.")
    else:
        print(f"WARNING: {learned_path} not found; skipping merge.")

# read the fresh breakdown to display and to keep accounting consistent
cat = pd.read_csv("artifacts/categories_breakdown.csv")
print("\n=== C) Machine-learned categories (French receipts) ===")
print(cat.to_string(index=False))
print(f"\nSum of category spend: {float(cat['spend'].sum()):.2f}")
print(f"Sum of all item totals : {float(it['line_total'].sum()):.2f}")

# persist learned mapping for later reuse in CI/pages
it[[name_col,"category_name"]].drop_duplicates().rename(columns={name_col:"product"}).to_csv(
    os.path.join(OUT_DIR, "categories_learned.csv"), index=False
)

# ======================================================
# D) Item price percentiles (unit_price)
# ======================================================
print("\n=== D) Item price distribution (unit price percentiles) ===")
it["unit_price"] = it["unit_price"].fillna(it["line_total"] / it["qty"])
prices = it["unit_price"].dropna().astype(float).values

if prices.size:
    percentiles = [10, 25, 50, 75, 90]
    pvals = np.percentile(prices, percentiles)
    for p, val in zip(percentiles, pvals):
        print(f"P{p:02}: {val:.2f} €")
    print(f"Min: {prices.min():.2f} €,  Max: {prices.max():.2f} €,  Mean: {prices.mean():.2f} €")
    pct_df = pd.DataFrame({"percentile": [f"P{p}" for p in percentiles], "value": pvals})
else:
    print("No prices found.")
    pct_df = pd.DataFrame(columns=["percentile","value"])

pct_df.to_csv(os.path.join(OUT_DIR, "price_percentiles.csv"), index=False)

# ======================================================
# HTML build (no charts) — writes artifacts/dashboard.html
# ======================================================
def _fmt_eur(x):
    try:
        v = float(x) if pd.notna(x) else 0.0
        return f"€{v:,.2f}".replace(",", " ")
    except Exception:
        return "€0.00"

def _table(df, cols, header=None, empty="No data"):
    if df is None or df.empty:
        return f"<em>{empty}</em>"
    use = [c for c in cols if c in df.columns]
    if not use:
        return f"<em>{empty}</em>"
    t = df.loc[:, use].copy()
    for c in t.columns:
        if any(k in c for k in ("total","price","spend","value")):
            t[c] = t[c].apply(_fmt_eur)
    if header and len(header) == len(use):
        t.columns = header
    return t.to_html(index=False, border=0, classes="tbl")

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
 .muted{color:#777;font-size:12px;margin-bottom:8px}
</style>
"""

items_total = float(it["line_total"].fillna(0).sum()) if "line_total" in it.columns else 0.0

parts = []
parts.append(f"<div class='muted'>Build: {VERSION_TAG}</div>")
# KPIs + reconciliation
parts.append(f"""
<h1>Naturalia — Spend Dashboard</h1>
<div class="kpis">
  <div class="kpi"><div class="label">Receipts</div><div class="value">{len(rc)}</div></div>
  <div class="kpi"><div class="label">Items</div><div class="value">{len(it)}</div></div>
  <div class="kpi"><div class="label">Receipts Total</div><div class="value">{_fmt_eur(grand_total)}</div></div>
</div>
<h3>Reconciliation (Items vs Receipts)</h3>
<table class="tbl">
  <tr><th>Source</th><th>Amount</th></tr>
  <tr><td>receipts_total</td><td>{_fmt_eur(grand_total)}</td></tr>
  <tr><td>items_total</td><td>{_fmt_eur(items_total)}</td></tr>
  <tr><td>delta</td><td>{_fmt_eur(items_total - grand_total)}</td></tr>
</table>
""")

# receipt-based time slices
parts.append("<h3>Spend by Month</h3>")
parts.append(_table(r_by_month, ["month","total"], ["Month","Spend"], "No dated receipts"))

parts.append("<h3>Spend by ISO Week</h3>")
parts.append(_table(r_by_week, ["week","total"], ["Week","Spend"], "No dated receipts"))

parts.append("<h3>Spend by Type of Day</h3>")
parts.append(_table(r_by_tod, ["type_of_day","total"], ["Type of Day","Spend"], "No dated receipts"))

parts.append("<h3>Spend by Weekday</h3>")
parts.append(_table(r_by_wday, ["weekday","total"], ["Weekday","Spend"], "No dated receipts"))

# price-evolution table
parts.append("<h3>Price Change (Last vs Previous)</h3>")
parts.append(_table(
    price_change,
    ["item","prev_date","prev_price","last_date","last_price","Δ_price","Δ_%","days_between"],
    ["Item","Prev Date","Prev Price","Last Date","Last Price","Δ Price","Δ %","Days Between"],
    "No repeated products yet"
))

# learned category spend
parts.append("<h3>Spend by Category (Learned)</h3>")
parts.append(_table(cat, ["category_name","spend"], ["Category","Spend"], "No learned categories yet"))

# percentiles
parts.append("<h3>Price Percentiles (Unit Price)</h3>")
parts.append(_table(pct_df, ["percentile","value"], ["Percentile","Value"], "No prices"))

html = (
    "<!doctype html><html><head><meta charset='utf-8'>"
    "<title>Naturalia — Dashboard</title>"
    + STYLE + "</head><body>" + "".join(parts) + "</body></html>"
)

# ======== FINAL SAFETY: always write artifacts/dashboard.html, independent of prior code ========
def __write_dashboard_html():
    import os, re, numpy as np, pandas as pd

    RCPT = "data/receipts.csv"
    ITEM = "data/items.csv"
    OUT_DIR = "artifacts"
    OUT_HTML = os.path.join(OUT_DIR, "dashboard.html")
    os.makedirs(OUT_DIR, exist_ok=True)

    # Load fresh
    rc = pd.read_csv(RCPT, low_memory=False)
    it = pd.read_csv(ITEM, low_memory=False)

    # Normalize
    rc["date"]  = pd.to_datetime(rc.get("date"), errors="coerce")
    rc["total"] = pd.to_numeric(rc.get("total"), errors="coerce").fillna(0)
    it["qty"]         = pd.to_numeric(it.get("qty"), errors="coerce").fillna(1)
    it["unit_price"]  = pd.to_numeric(it.get("unit_price"), errors="coerce")
    it["line_total"]  = pd.to_numeric(it.get("line_total"), errors="coerce")

    # Receipt-based time slices
    rc_ok = rc.dropna(subset=["date"]).copy()
    grand_total = float(rc_ok["total"].sum())
    r_by_month = (rc_ok.assign(month=rc_ok["date"].dt.to_period("M").astype(str))
                      .groupby("month", as_index=False)["total"].sum().sort_values("month"))
    r_by_week  = (rc_ok.assign(week=rc_ok["date"].dt.strftime("%G-W%V"))
                      .groupby("week",  as_index=False)["total"].sum().sort_values("week"))
    r_by_tod   = (rc_ok.assign(type_of_day=rc_ok["date"].dt.weekday.map(lambda d: "Weekend" if d>=5 else "Weekday"))
                      .groupby("type_of_day", as_index=False)["total"].sum())
    r_by_wday  = (rc_ok.assign(weekday=rc_ok["date"].dt.day_name())
                      .groupby("weekday", as_index=False)["total"].sum())

    # Percentiles (derive now if not already persisted)
    it["unit_price"] = it["unit_price"].fillna(it["line_total"]/it["qty"])
    prices = it["unit_price"].dropna().astype(float).values
    if prices.size:
        pcts   = [10, 25, 50, 75, 90]
        pvals  = np.percentile(prices, pcts)
        pct_df = pd.DataFrame({"percentile":[f"P{p}" for p in pcts], "value":pvals})
    else:
        pct_df = pd.DataFrame(columns=["percentile","value"])

    # Learned categories: expect artifacts/categories_learned.csv = (product, category_name)
    cat_map_path = os.path.join(OUT_DIR, "categories_learned.csv")
    cat = pd.DataFrame(columns=["category_name","spend"])
    name_col = "product" if "product" in it.columns else ("product_norm" if "product_norm" in it.columns else None)
    if name_col and os.path.exists(cat_map_path) and os.path.getsize(cat_map_path) > 0:
        catmap = pd.read_csv(cat_map_path)
        right_key = "product" if "product" in catmap.columns else name_col
        merged = it.merge(
            catmap.rename(columns={ right_key: name_col }),
            on=name_col, how="left"
        )
        merged["category_name"] = merged["category_name"].fillna("other")
        cat = (merged.groupby("category_name", as_index=False)["line_total"]
                     .sum().rename(columns={"line_total":"spend"})
                     .sort_values("spend", ascending=False))

    # Price change table: optional placeholder (the main script can compute a real one)
    price_change = globals().get("price_change", pd.Data120Frame()) if "pd" in globals() else pd.DataFrame()
    if price_change.empty or not set(["item","prev_date","prev_price","last_date","last_price","Δ_price","Δ_%","days_between"]).issubset(price_change.columns):
        price_change = pd.DataFrame(columns=["item","prev_date","prev_price","last_date","last_price","Δ_price","Δ_%","days_between"])

    # Helpers
    def fmt_eur(x):
        try:
            v = float(x) if pd.notna(x) else 0.0
            return f"€{v:,.2f}".replace(",", " ")
        except Exception:
            return "€0.00"

    def table(df, cols, header=None, empty="No data"):
        if df is None or df.empty:
            return f"<em>{empty}</em>"
        use = [c for c in cols if c in df.columns]
        if not use:
            return f"<em>{empty}</em>"
        t = df.loc[:, use].copy()
        for c in t.columns:
            if any(k in c for k in ("total","price","spend","value")):
                t[c] = t[c].apply(fmt_eur)
        if header and len(header) == len(use):
            t.columns = header
        return t.to_html(index=False, border=0, classes="tbl")

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
    parts.append(f"""
    <h1>Naturalia — Spend Dashboard</h1>
    <div class="kpis">
      <div class="kpi"><div class="label">Receipts</div><div class="value">{len(rc)}</div></div>
      <div class="kpi"><div class="label">Items</div><div class="value">{len(it)}</div></div>
      <div class="kpi"><div class="label">Receipts Total</div><div class="value">{fmt_eur(grand_total if 'grand_total' in globals() else grand_total)}</div></div>
    </div>
    <h3>Reconciliation (Items vs Receipts)</h3>
    <table class="tbl">
      <tr><th>Source</th><th>Amount</th></tr>
      <tr><td>receipts_total</td><td>{fmt_eur(grand_total)}</td></tr>
      <tr><td>items_total</td><td>{fmt_eur(items_total)}</td></tr>
      <tr><td>delta</td><td>{fmt_eur(items_total - grand_total)}</td></tr>
    </table>
    """)

    # Receipt-based slices
    parts.append("<h3>Spend by Month</h3>");      parts.append(table(r_by_month, ["month","total"], ["Month","Spend"]))
    parts.append("<h3>Spend by ISO Week</h3>");   parts.append(table(r_by_week,  ["week","total"], ["Week","Spend"]))
    parts.append("<h3>Spend by Type of Day</h3>");parts.append(table(r_by_tod,   ["type_of_day","total"], ["Type of Day","Spend"]))
    parts.append("<h3>Spend by Weekday</h3>");    parts.append(table(r_by_wday,  ["weekday","total"], ["Weekday","Spend"]))

    # Price change (if available)
    parts.append("<h3>Price Change (Last vs Previous)</h3>")
    parts.append(table(
        price_change,
        ["item","prev_date","prev_price","last_date","last_price","Δ_price","Δ_%","days_between"],
        ["Item","Prev Date","Prev Price","Last Date","Last Price","Δ Price","Δ %","Days Between"],
        "No repeated products yet"
    ))

    # Categories (spend)
    parts.append("<h3>Spend by Category (Learned)</h3>")
    parts.append(table(cat, ["category_name","spend"], ["Category","Spend"], "No learned categories yet"))

    # Percentiles
    parts.append("<h3>Price Percentiles (Unit Price)</h3>")
    parts.append(table(pct_df, ["percentile","value"], ["Percentile","Value"], "No prices"))

    html = "<!doctype html><html><head><meta charset='utf-8'><title>Naturalia — Dashboard</title>" + STYLE + "</head><body>" + "".join(parts) + "</body></html>"
    with open(OUT_HTML, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"\nWrote {os.path.abspath(OUT_HTML)}")

# Call the writer unconditionally so we always have a file
try:
    __write_dashboard_html()
except Exception as __e:
    print("HTML write failed:", __e)
