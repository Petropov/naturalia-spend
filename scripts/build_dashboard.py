# scripts/build_dashboard.py
import os
import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import numpy as np

# ---------- Load & prepare data ----------
RCPT = "data/receipts.csv"
ITEM = "data/items.csv"
OUT_DIR = "artifacts"
os.makedirs(OUT_DIR, exist_ok=True)

rc = pd.read_csv(RCPT, low_memory=False)
it = pd.read_csv(ITEM, low_memory=False)

# normalize types
rc["date"]  = pd.to_datetime(rc.get("date"), errors="coerce")
rc["total"] = pd.to_numeric(rc.get("total"), errors="coerce").fillna(0)
it["qty"]        = pd.to_numeric(it.get("qty"), errors="coerce").fillna(1)
it["unit_price"] = pd.to_numeric(it.get("unit_price"), errors="coerce")
it["line_total"] = pd.to_numeric(it.get("line_total"), errors="coerce")

print("== Naturalia spend analysis ==")
print(f"Receipts: {len(rc)}, Items: {len(it)}")

# ==============================================================
#  A) Time-slice totals (must match grand total)
# ==============================================================
rc_valid = rc.dropna(subset=["date"]).copy()
grand_total = rc_valid["total"].sum()

by_month = rc_valid.assign(month=rc_valid["date"].dt.to_period("M").astype(str)) \
                   .groupby("month", as_index=False)["total"].sum()
by_week  = rc_valid.assign(week=rc_valid["date"].dt.strftime("%G-W%V")) \
                   .groupby("week", as_index=False)["total"].sum()
by_wday  = rc_valid.assign(weekday=rc_valid["date"].dt.day_name()) \
                   .groupby("weekday", as_index=False)["total"].sum()

print("\n=== A) Totals by time slices ===")
print(f"Grand total: {grand_total:.2f}")
print(f"Sum by month: {by_month['total'].sum():.2f}")
print(f"Sum by week : {by_week ['total'].sum():.2f}")
print(f"Sum by wday : {by_wday ['total'].sum():.2f}\n")
print("-- by month --")
print(by_month.to_string(index=False))
print("-- by week --")
print(by_week.to_string(index=False))
print("-- by weekday --")
print(by_wday.to_string(index=False))


# === B) Price evolution across receipts (warning-free) ===

# 1) pick the item name column that exists
name_col = (
    "product" if "product" in it.columns else
    ("product_norm" if "product_norm" in it.columns else
     ("product_raw" if "product_raw" in it.columns else None))
)
if name_col is None:
    name_col = "product"  # fallback
    it[name_col] = ""

# 2) join items -> receipt date with a stable key
join_key = "receipt_uid" if "receipt_uid" in it.columns else "receipt_id"
tmp = it.copy()
tmp["qty"]        = pd.to_numeric(tmp.get("qty"), errors="coerce").fillna(1)
tmp["unit_price"] = pd.to_numeric(tmp.get("unit_price"), errors="coerce")
tmp["line_total"] = pd.to_numeric(tmp.get("line_total"), errors="coerce")
tmp["unit_price"] = tmp["unit_price"].fillna(tmp["line_total"] / tmp["qty"])

rc_join = rc_valid[[join_key, "date"]].rename(columns={join_key: "receipt_id"})
tmp = tmp.merge(rc_join, on="receipt_id", how="left")
tmp["date"] = pd.to_datetime(tmp["date"], errors="coerce")

# keep only items that appear in at least 2 different receipts
counts = tmp.groupby(name_col)["receipt_id"].nunique()
repeated_names = counts[counts >= 2].index
tmp = tmp[tmp[name_col].isin(repeated_names)]

# if nothing repeats, short-circuit cleanly
if tmp.empty:
    price_change = pd.DataFrame(
        columns=["item","prev_date","prev_price","last_date","last_price","Δ_price","Δ_%","days_between"]
    )
else:
    # 3) compute previous & last rows per item WITHOUT groupby.apply
    s = tmp.sort_values([name_col, "date"])

    # last occurrence per item
    idx_last = s.groupby(name_col)["date"].idxmax()
    last = s.loc[idx_last, [name_col, "date", "unit_price"]].rename(
        columns={"date": "last_date", "unit_price": "last_price"}
    )

    # previous occurrence per item: drop the last row then take max again
    s_wo_last = s.drop(index=idx_last)
    idx_prev = s_wo_last.groupby(name_col)["date"].idxmax()
    prev = s_wo_last.loc[idx_prev, [name_col, "date", "unit_price"]].rename(
        columns={"date": "prev_date", "unit_price": "prev_price"}
    )

    # 4) combine & compute deltas
    pc = prev.merge(last, on=name_col, how="inner")
    pc["Δ_price"] = (pc["last_price"] - pc["prev_price"]).round(2)
    pc["Δ_%"] = ((pc["last_price"] - pc["prev_price"]) / pc["prev_price"] * 100).round(2)
    pc["days_between"] = (pd.to_datetime(pc["last_date"]) - pd.to_datetime(pc["prev_date"])).dt.days

    price_change = pc.rename(columns={name_col: "item"})[
        ["item","prev_date","prev_price","last_date","last_price","Δ_price","Δ_%","days_between"]
    ].sort_values(["Δ_%","Δ_price"], ascending=[False, False]).reset_index(drop=True)


# ==============================================================
#  C) Category discovery (French-aware unsupervised clustering)
# ==============================================================
print("\n=== C) Machine-learned categories (French receipts) ===")

def clean_fr(s: str) -> str:
    s = (s or "").lower()
    s = re.sub(r"\d+[x×]?\s*\d*(ml|l|g|kg|cl|pack|pcs)?", " ", s)
    s = re.sub(r"[^a-zàâäéèêëîïôöùûüç\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

it["clean"] = it["product"].fillna("").map(clean_fr)

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
print(f"\nSum of category spend: {cat['spend'].sum():.2f}")
print(f"Sum of all item totals : {it['line_total'].sum():.2f}")

print("\n-- Example items per cluster --")
for name, sub in it.groupby("category_name"):
    ex = ", ".join(sub["product"].head(5))
    print(f"{name}: {ex}")

it[["product","category_name"]].drop_duplicates().to_csv(
    os.path.join(OUT_DIR, "categories_learned.csv"), index=False)

# ==============================================================
#  D) Item price percentiles
# ==============================================================
print("\n=== D) Item price distribution (unit price percentiles) ===")

# derive per-unit prices
it["unit_price"] = it["unit_price"].fillna(it["line_total"]/it["qty"])
prices = it["unit_price"].dropna()
percentiles = [10,25,50,75,90]
p_values = np.percentile(prices, percentiles)

for p, val in zip(percentiles, p_values):
    print(f"P{p:02}: {val:.2f} €")

print(f"Min: {prices.min():.2f} €,  Max: {prices.max():.2f} €,  Mean: {prices.mean():.2f} €")

# Save percentile summary to artifacts
pd.DataFrame({"percentile": [f"P{p}" for p in percentiles],
              "value": p_values}).to_csv(
    os.path.join(OUT_DIR,"price_percentiles.csv"), index=False)

print("\nAnalysis complete. Outputs saved to 'artifacts/'.")
