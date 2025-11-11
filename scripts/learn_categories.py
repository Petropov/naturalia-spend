# scripts/learn_categories.py
import os, re, json, joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score

ART = Path("artifacts")
ART.mkdir(exist_ok=True)

ITEMS = Path("data/items.csv")
MAP_OUT   = ART / "categories_learned.csv"
BREAK_OUT = ART / "categories_breakdown.csv"
META_OUT  = ART / "categories_meta.json"
VECT_OUT  = ART / "tfidf.pkl"
KM_OUT    = ART / "kmeans.pkl"

def clean_fr(s: str) -> str:
    s = (s or "").lower()
    s = re.sub(r"\d+[x×]?\s*\d*(ml|l|g|kg|cl|pack|pcs)?", " ", s)
    s = re.sub(r"[^a-zàâäéèêëîïôöùûüç\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def top_terms_for_center(center, terms, n=3):
    idx = np.argsort(center)[-n:][::-1]
    return "|".join(terms[i] for i in idx)

def pick_k_by_consensus(X, k_grid):
    best = None
    best_score = -1
    evals = []
    for k in k_grid:
        km = KMeans(n_clusters=k, n_init="auto", random_state=42).fit(X)
        lab = km.labels_
        # guard: silhouette needs >1 label
        if len(set(lab)) < 2:
            s = -1
            db = np.inf
        else:
            s  = silhouette_score(X, lab, sample_size=min(2000, X.shape[0]), random_state=42)
            db = davies_bouldin_score(X.toarray() if hasattr(X, "toarray") else X, lab)
        # simple consensus: higher silhouette, lower DB
        score = s - (db/10.0)
        evals.append({"k":k, "sil":float(s), "db":float(db), "score":float(score)})
        if score > best_score:
            best_score = score; best = (k, km)
    return best, evals

def main():
    if not ITEMS.exists():
        print("No items.csv found.")
        return

    it = pd.read_csv(ITEMS, low_memory=False)
    # name column
    name_col = "product" if "product" in it.columns else ("product_norm" if "product_norm" in it.columns else None)
    if not name_col:
        print("No product/product_norm column in items.csv")
        return

    it[name_col]   = it[name_col].fillna("").astype(str)
    it["clean"]    = it[name_col].map(clean_fr)
    it["line_total"] = pd.to_numeric(it.get("line_total"), errors="coerce").fillna(0)

    # vectorizer: reuse if present (stability), else fit
    if VECT_OUT.exists():
        vect = joblib.load(VECT_OUT)
        X = vect.transform(it["clean"])
        terms = vect.get_feature_names_out()
    else:
        stop_fr = {"de","du","des","le","la","les","un","une","et","ou","en","au","aux","pour","avec","sans","sur","sous","dans","à","par","chez","tout","tous"}
        vect = TfidfVectorizer(min_df=1, ngram_range=(1,2), stop_words=list(stop_fr))
        X = vect.fit_transform(it["clean"])
        terms = vect.get_feature_names_out()
        joblib.dump(vect, VECT_OUT)

    # choose K by consensus (5..min(15, N//5)); keep previous model if loadable and N similar
    N = it.shape[0]
    k_grid = list(range(5, max(6, min(15, max(6, N//5)))+1))
    reuse = KM_OUT.exists()
    prev_km = joblib.load(KM_OUT) if reuse else None

    if prev_km is not None and prev_km.n_clusters in k_grid:
        km = prev_km.fit(X)
        k  = km.n_clusters
        evals = []
    else:
        (k, km), evals = pick_k_by_consensus(X, k_grid)
        joblib.dump(km, KM_OUT)

    labs  = km.labels_
    cents = km.cluster_centers_

    # auto-name clusters (spend-weighted re-ranking of term relevance)
    base_names = [top_terms_for_center(c, terms, n=5) for c in cents]
    names = []
    df = pd.DataFrame({"name": it[name_col], "clean": it["clean"], "lab": labs, "w": it["line_total"]})
    for cid in range(k):
        sub = df[df["lab"]==cid]
        if sub.empty:
            names.append(base_names[cid].split("|")[:3])
            continue
        # term scores inside cluster weighted by spend
        toks = " ".join(sub["clean"]).split()
        if not toks:
            names.append(base_names[cid].split("|")[:3]); continue
        u, c = np.unique(toks, return_counts=True)
        # weight by spend proxy (mean spend in cluster)
        weight = sub["w"].mean() if sub["w"].notna().any() else 1.0
        idx = np.argsort(c * weight)[-3:][::-1]
        names.append(u[idx].tolist())
    label_map = {i: " • ".join([t for t in names[i] if t][:3]) for i in range(k)}
    # fallback to base tf-idf names if empty
    for i in range(k):
        if not label_map[i].strip():
            label_map[i] = " • ".join(base_names[i].split("|")[:3])

    it["category_id"]   = labs
    it["category_name"] = it["category_id"].map(label_map)

    # outputs
    mapping = it[[name_col, "category_name"]].drop_duplicates().rename(columns={name_col:"product"})
    mapping.to_csv(MAP_OUT, index=False)

    breakdown = it.groupby("category_name", as_index=False)["line_total"].sum().rename(columns={"line_total":"spend"}) \
                 .sort_values("spend", ascending=False)
    breakdown.to_csv(BREAK_OUT, index=False)

    meta = {
        "n_items": int(N),
        "k": int(k),
        "evals": evals,
        "label_examples": {str(i): label_map[i] for i in range(k)}
    }
    META_OUT.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Wrote {MAP_OUT}, {BREAK_OUT}, {META_OUT}")

if __name__ == "__main__":
    main()
