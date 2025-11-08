from pathlib import Path
import re, pandas as pd, numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sentence_transformers import SentenceTransformer
from collections import Counter

DATA   = Path("data")
REPORT = Path("reports/items_clusters.html")
REPORT.parent.mkdir(exist_ok=True)

items = pd.read_csv(DATA/"items.csv")
items = items[["receipt_id","product","qty","unit_price","line_total","currency"]].copy()
items["product"] = items["product"].fillna("").astype(str)

def norm(txt:str)->str:
    t = txt.lower()
    t = re.sub(r"\b\d+(?:g|cl|ml)\b"," ", t)     # drop sizes
    t = re.sub(r"\b\d+x\b"," ", t)               # drop leading multipliers
    t = re.sub(r"[^a-zàâçéèêëîïôùûüÿñæœ\s]", " ", t)
    t = re.sub(r"\s+"," ", t).strip()
    return t

prod = items["product"].unique()
prod_norm = [norm(p) for p in prod]

if len(prod) == 0:
    REPORT.write_text("<h1>No products found</h1>", encoding="utf-8")
    print("No products to cluster.")
    raise SystemExit(0)

model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
X = model.encode(prod_norm, show_progress_bar=False, normalize_embeddings=True)

def choose_k(X):
    n = len(X)
    if n <= 2: return n
    cand = [k for k in range(2, min(8, n))]
    best_k, best_s = 2, -1
    for k in cand:
        km = KMeans(n_clusters=k, n_init="auto", random_state=0).fit(X)
        s = silhouette_score(X, km.labels_) if k < n else -1
        if s > best_s: best_s, best_k = s, k
    return best_k

k = choose_k(X)
if k <= 1: k = 1
km = KMeans(n_clusters=k, n_init="auto", random_state=0).fit(X)
labels = km.labels_

cluster_of = {p: int(lbl) for p,lbl in zip(prod, labels)}
items["cluster_id"] = items["product"].map(cluster_of)

def top_terms(names, top_n=3):
    tokens = []
    for n in names:
        tokens.extend([w for w in norm(n).split() if len(w) >= 3])
    if not tokens: return "Misc"
    commons = [w for (w,_) in Counter(tokens).most_common(5)]
    return " / ".join(commons[:top_n]).title()

cluster_labels = {}
for cid in sorted(set(labels)):
    names = [p for p,l in zip(prod, labels) if l==cid]
    cluster_labels[cid] = top_terms(names)

items["cluster_label"] = items["cluster_id"].map(cluster_labels)

by_cluster = (items
    .groupby(["cluster_id","cluster_label"], as_index=False)["line_total"].sum()
    .sort_values("line_total", ascending=False))

by_product = (items
    .groupby(["cluster_id","product"], as_index=False)
    .agg(qty=("qty","sum"), spend=("line_total","sum"))
    .sort_values(["cluster_id","spend"], ascending=[True, False]))

money = lambda x: "€{:,.2f}".format(x).replace(",", " ")
by_cluster["line_total"] = by_cluster["line_total"].map(money)
by_product["spend"] = by_product["spend"].map(money)

def table(df, title):
    if df.empty: return f"<h2>{title}</h2><p class='muted'>No data.</p>"
    return f"<h2>{title}</h2>" + df.to_html(index=False, escape=False)

blocks = []
blocks.append(table(by_cluster.rename(columns={"line_total":"Spend (€)","cluster_label":"Cluster"}), "Clusters — total spend"))
for cid, grp in by_product.groupby("cluster_id"):
    label = cluster_labels.get(cid, f"Cluster {cid}")
    blocks.append(table(grp.drop(columns=["cluster_id"]).rename(columns={"product":"Product","qty":"Qty","spend":"Spend (€)"}), f"Cluster {cid}: {label}"))

html = f"""
<!doctype html><meta charset="utf-8">
<title>Naturalia — Item Clusters</title>
<style>
body{{font-family:-apple-system,BlinkMacSystemFont,Segoe UI,Inter,Roboto,Helvetica,Arial,sans-serif;margin:24px}}
h1{{margin:0 0 8px}} h2{{margin:18px 0 10px}} .muted{{color:#666}}
table{{border-collapse:collapse;width:100%;background:#fff;margin:8px 0 16px}}
th,td{{border:1px solid #e7e7e7;padding:8px 10px;text-align:left}}
th{{background:#fafafa}}
</style>
<h1>Naturalia — Item Clusters</h1>
<div class="muted">k = {k} • {len(prod)} unique products • {len(items)} lines</div>
{''.join(blocks)}
"""
REPORT.write_text(html, encoding="utf-8")
print("Wrote", REPORT)
