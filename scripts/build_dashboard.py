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

# ==============================================================
#  B) Price change for repeated products
# ==============================================================
print("\n=== B) Price evolution across receipts ===")

join_key = "receipt_uid" if "receipt_uid" in it.columns else "receipt_id"
df = it.merge(rc_valid[[join_key,"date"]].rename(columns={join_key:"receipt_id"}), 
              on="receipt_id", how="left")

df["unit_price"] = df["unit_price"].fillna(df["line_total"]/df["qty"])
df["date"] = pd.to_datetime(df["date"], errors="coerce")

name_col = "product" if "product" in df.columns else \
           ("product_norm" if "product_norm" in df.columns else df.columns[2])

multi = df.groupby(name_col).filter(lambda g: g["receipt_id"].nunique() > 1)

def price_change(g):
    g = g.sort_values("date")
    if len(g) < 2:
        return None
    prev, last = g.iloc[-2], g.iloc[-1]
    return pd.Series({
        "product": g.name,
        "prev_date": prev["date"].date(),
        "prev_price": round(prev["unit_price"],2),
        "last_date": last["date"].date(),
        "last_price": round(last["unit_price"],2),
        "Δ_price": round(last["unit_price"]-prev["unit_price"],2),
        "Δ_%": round((last["unit_price"]-prev["unit_price"])/prev["unit_price"]*100,2)
                if prev["unit_price"] else None
    })

changes = multi.groupby(name_col).apply(price_change).dropna().reset_index(drop=True)
if len(changes)==0:
    print("No repeated product names across receipts yet.")
else:
    print(changes.to_string(index=False))

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
