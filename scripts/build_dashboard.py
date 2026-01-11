# scripts/build_dashboard.py
import os
import numpy as np
import pandas as pd
from pathlib import Path
from rapidfuzz import fuzz
from scripts.text_utils import canonical_product_key, normalize_text

# ------------------ config / io ------------------
VERSION_TAG = "Naturalia Dashboard v3 — receipts-only A, warning-free B, ML-C (fr), P10..P90, HTML"
RCPT = "data/receipts.csv"
ITEM = "data/items.csv"
OUT_DIR = "artifacts"
OUT_HTML = os.path.join(OUT_DIR, "dashboard.html")
Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

print("== Naturalia spend analysis ==")

def pick_price(row) -> float | None:
    for key in ("unit_price",):
        val = pd.to_numeric(row.get(key), errors="coerce")
        if pd.notna(val) and val > 0:
            return float(val)
    line_total = pd.to_numeric(row.get("line_total"), errors="coerce")
    qty = pd.to_numeric(row.get("qty"), errors="coerce")
    if pd.notna(line_total) and pd.notna(qty) and qty > 0:
        val = line_total / qty
        if pd.notna(val) and val > 0:
            return float(val)
    if pd.notna(line_total) and line_total > 0:
        return float(line_total)
    return None

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

name_col = (
    "product" if "product" in it.columns else
    ("product_norm" if "product_norm" in it.columns else
     ("product_raw" if "product_raw" in it.columns else None))
)
if name_col is None:
    name_col = "product"
    it[name_col] = ""

print(f"Item name column: {name_col}")

join_key = None
for cand in ("receipt_uid", "receipt_id"):
    if cand in it.columns and cand in rc_valid.columns:
        join_key = cand
        break

it["product_key"] = it[name_col].astype(str).map(canonical_product_key)
it_pc = it[it["product_key"].str.len() > 0].copy()
it_pc["price_eff"] = it_pc.apply(pick_price, axis=1)

if join_key is not None:
    rc_map = (
        rc_valid.sort_values("date")
        .drop_duplicates(subset=[join_key], keep="first")
        [[join_key, "date"]]
    )
    it_pc = it_pc.merge(rc_map, on=join_key, how="left")
else:
    it_pc["date"] = pd.NaT
    print("No join key between items and receipts; skipping price-change by date.")

it_pc["date"] = pd.to_datetime(it_pc["date"], errors="coerce")

mapped_pct = float(it_pc["date"].notna().mean() * 100) if len(it_pc) else 0.0
repeat_counts = it_pc.groupby("product_key").size()
repeated_product_keys = repeat_counts[repeat_counts >= 2]
repeats_with_dates = (
    it_pc.groupby("product_key")["date"].nunique().reindex(repeated_product_keys.index).fillna(0) >= 2
).sum()

print(f"Receipts rows: {len(rc)}")
print(f"Items rows: {len(it)}")
print(f"Join key: {join_key or 'none'}")
print(f"Items mapped to receipt date: {mapped_pct:.1f}%")
print(f"Repeated product keys (>=2 items): {int(repeated_product_keys.shape[0])}")
print(f"Repeated product keys with >=2 dates: {int(repeats_with_dates)}")

fuzzy_threshold = 92
use_fuzzy = repeated_product_keys.empty and len(it_pc) <= 2000
if use_fuzzy:
    name_norm = it_pc[name_col].astype(str).map(lambda s: normalize_text(s, stopwords=None))
    unique_names = sorted(set(name_norm) - {""})
    clusters = []
    assignment = {}
    scores = {}
    for name in unique_names:
        best_idx = None
        best_score = -1
        for idx, rep in enumerate(clusters):
            score = fuzz.token_sort_ratio(name, rep)
            if score >= fuzzy_threshold and score > best_score:
                best_idx = idx
                best_score = score
        if best_idx is None:
            clusters.append(name)
            assignment[name] = len(clusters) - 1
            scores[name] = 100
        else:
            assignment[name] = best_idx
            scores[name] = best_score
    it_pc["product_key"] = name_norm.map(lambda n: f"fuzzy_{assignment.get(n, 0):03d}")
    it_pc["match_confidence"] = name_norm.map(lambda n: f"{scores.get(n, 0):.0f}")
else:
    it_pc["match_confidence"] = "exact"

eligible = it_pc.dropna(subset=["date"]).copy()
eligible = eligible[eligible["price_eff"].notna()]
counts = eligible.groupby("product_key").agg(
    n_rows=("product_key", "size"),
    n_dates=("date", "nunique"),
    n_prices=("price_eff", "count"),
)
eligible_keys = counts[(counts["n_rows"] >= 2) & (counts["n_dates"] >= 2) & (counts["n_prices"] >= 2)].index
eligible = eligible[eligible["product_key"].isin(eligible_keys)]

if eligible.empty:
    price_change = pd.DataFrame(
        columns=["item","prev_date","prev_price","last_date","last_price","Δ_price","Δ_%","days_between","match_confidence"]
    )
else:
    s = eligible.sort_values(["product_key", "date"])
    idx_last = s.groupby("product_key")["date"].idxmax()
    last = (
        s.loc[idx_last, ["product_key", "date", "price_eff", "match_confidence"]]
         .rename(columns={"date": "last_date", "price_eff": "last_price"})
    )
    s_wo_last = s.drop(index=idx_last)
    idx_prev = s_wo_last.groupby("product_key")["date"].idxmax()
    prev = (
        s_wo_last.loc[idx_prev, ["product_key", "date", "price_eff"]]
                 .rename(columns={"date": "prev_date", "price_eff": "prev_price"})
    )
    pc = prev.merge(last, on="product_key", how="inner")
    pc["Δ_price"] = (pc["last_price"] - pc["prev_price"]).round(2)
    pc["Δ_%"] = ((pc["last_price"] - pc["prev_price"]) / pc["prev_price"] * 100).round(2)
    pc["days_between"] = (
        pd.to_datetime(pc["last_date"]) - pd.to_datetime(pc["prev_date"])
    ).dt.days
    display_names = (
        it_pc.groupby("product_key")[name_col]
        .agg(lambda x: x.value_counts().index[0] if not x.empty else "")
        .to_dict()
    )
    pc["item"] = pc["product_key"].map(display_names).fillna(pc["product_key"])
    price_change = (
        pc[["item","prev_date","prev_price","last_date","last_price","Δ_price","Δ_%","days_between","match_confidence"]]
          .sort_values(["Δ_%","Δ_price"], ascending=[False, False])
          .reset_index(drop=True)
    )

repeat_debug = pd.DataFrame()
if repeated_product_keys.any() or use_fuzzy:
    summary = it_pc.groupby("product_key").agg(
        count=("product_key", "size"),
        distinct_dates=("date", "nunique"),
        non_null_prices=("price_eff", lambda s: s.notna().sum()),
        example_names=(name_col, lambda s: ", ".join(s.dropna().astype(str).unique()[:3])),
    ).sort_values("count", ascending=False)
    repeat_debug = summary.head(20).reset_index()

if not price_change.empty:
    print(price_change.to_string(index=False))
elif repeated_product_keys.any():
    if it_pc["date"].notna().sum() == 0:
        print("Repeated products detected, but cannot compute price change (missing receipt dates).")
    elif it_pc["price_eff"].notna().sum() == 0:
        print("Repeated products detected, but cannot compute price change (missing prices).")
    else:
        print("Repeated products detected, but cannot compute price change (insufficient date/price history).")
 
# ======================================================
# C) Machine-learned categories (French-aware) + spend by category
# ======================================================
from scripts.learn_categories import main as learn_cats

learn_cats()  # writes artifacts/*category* files

# Merge learned categories back into items (it)
if "category_name" not in it.columns:
    learned_path = os.path.join(OUT_DIR, "categories_learned.csv")
    if os.path.exists(learned_path):
        learned = pd.read_csv(learned_path)
        if "product_key" in learned.columns and "category_name" in learned.columns:
            it = it.merge(
                learned[["product_key", "category_name", "category_source"]].drop_duplicates(),
                on="product_key",
                how="left",
            )
        elif "product" in learned.columns and "category_name" in learned.columns:
            it = it.merge(
                learned[["product", "category_name"]].drop_duplicates(),
                left_on=name_col,
                right_on="product",
                how="left",
            ).drop(columns=["product"], errors="ignore")
        else:
            print(f"WARNING: {learned_path} missing expected columns; skipping merge.")
    else:
        print(f"WARNING: {learned_path} not found; skipping merge.")

it["category_name"] = it.get("category_name").fillna("Uncategorized")
it["category_source"] = it.get("category_source").fillna("uncategorized")

# read the fresh breakdown to display and to keep accounting consistent
cat = pd.read_csv("artifacts/categories_breakdown.csv")
print("\n=== C) Machine-learned categories (French receipts) ===")
print(cat.to_string(index=False))
print(f"\nSum of category spend: {float(cat['spend'].sum()):.2f}")
print(f"Sum of all item totals : {float(it['line_total'].sum()):.2f}")

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
        if "share" in c:
            t[c] = t[c].apply(lambda v: f"{(float(v) * 100):.1f}%" if pd.notna(v) else "0.0%")
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
price_change_empty_message = "No repeated products yet."
if repeated_product_keys.any() and price_change.empty:
    price_change_empty_message = "Repeated products detected, but cannot compute price change (missing dates/prices)."
elif join_key is None and price_change.empty:
    price_change_empty_message = "No join key between items and receipts; price change unavailable."
parts.append(_table(
    price_change,
    ["item","prev_date","prev_price","last_date","last_price","Δ_price","Δ_%","days_between","match_confidence"],
    ["Item","Prev Date","Prev Price","Last Date","Last Price","Δ Price","Δ %","Days Between","Match"],
    price_change_empty_message
))

if price_change.empty:
    parts.append("<h3>Repeat diagnostics</h3>")
    parts.append(
        f"<div class='muted'>Join key: {join_key or 'none'} · "
        f"Mapped to receipt date: {mapped_pct:.1f}% · "
        f"Repeated product keys: {int(repeated_product_keys.shape[0])} · "
        f"Repeated with ≥2 dates: {int(repeats_with_dates)}</div>"
    )
    parts.append(_table(
        repeat_debug,
        ["product_key","count","distinct_dates","non_null_prices","example_names"],
        ["Product Key","Count","Distinct Dates","Non-null Prices","Examples"],
        "No repeat diagnostics available."
    ))

# learned category spend
parts.append("<h3>Spend by Category (Learned)</h3>")
parts.append(_table(cat, ["category_name","spend"], ["Category","Spend"], "No learned categories yet"))

category_spend_total = float(it["line_total"].fillna(0).sum()) if "line_total" in it.columns else 0.0
coverage = (
    it.groupby("category_source", as_index=False)["line_total"].sum()
      .rename(columns={"line_total": "spend", "category_source": "source"})
)
coverage["share"] = coverage["spend"] / category_spend_total if category_spend_total else 0
coverage = coverage.sort_values("spend", ascending=False)

uncat_top = (
    it[it["category_name"] == "Uncategorized"]
      .groupby(name_col, as_index=False)["line_total"].sum()
      .sort_values("line_total", ascending=False)
      .head(10)
      .rename(columns={name_col: "item", "line_total": "spend"})
)

parts.append("<h3>Category Coverage</h3>")
parts.append(_table(
    coverage,
    ["source","spend","share"],
    ["Source","Spend","Share"],
    "No coverage data."
))
parts.append("<h3>Top Uncategorized Items by Spend</h3>")
parts.append(_table(
    uncat_top,
    ["item","spend"],
    ["Item","Spend"],
    "No uncategorized items."
))

# percentiles
parts.append("<h3>Price Percentiles (Unit Price)</h3>")
parts.append(_table(pct_df, ["percentile","value"], ["Percentile","Value"], "No prices"))

html = (
    "<!doctype html><html><head><meta charset='utf-8'>"
    "<title>Naturalia — Dashboard</title>"
    + STYLE + "</head><body>" + "".join(parts) + "</body></html>"
)
with open(OUT_HTML, "w", encoding="utf-8") as f:
    f.write(html)
print(f"\nWrote {os.path.abspath(OUT_HTML)}")
