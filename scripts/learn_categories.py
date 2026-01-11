# scripts/learn_categories.py
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from scripts.text_utils import canonical_product_key, normalize_text, STOPWORDS

ART = Path("artifacts")
ART.mkdir(exist_ok=True)

ITEMS = Path("data/items.csv")
MAP_OUT   = ART / "categories_learned.csv"
BREAK_OUT = ART / "categories_breakdown.csv"
META_OUT  = ART / "categories_meta.json"

RULE_LEXICON = {
    "Dairy": {"lait", "yaourt", "fromage", "beurre", "creme", "mozza", "mozzarella"},
    "Produce": {"avocat", "tomate", "salade", "pomme", "banane", "citron", "oignon"},
    "Staples": {"riz", "pate", "farine", "sucre", "sel", "huile"},
    "Drinks": {"kombucha", "eau", "jus", "vin", "biere", "gin"},
    "Condiments": {"shoyu", "sauce", "vinaigre", "moutarde"},
    "Snacks": {"chocolat", "chips", "biscuits"},
    "Protein": {"tofu", "poulet", "boeuf", "poisson", "oeuf"},
}

RULE_ORDER = list(RULE_LEXICON.keys())
CLUSTER_STOPWORDS = STOPWORDS | {
    "de","du","des","le","la","les","un","une","et","ou","en","au","aux",
    "pour","avec","sans","sur","sous","dans","par","chez","tout","tous",
}

def rule_category(tokens: set[str]) -> str | None:
    for label in RULE_ORDER:
        if tokens & RULE_LEXICON[label]:
            return label
    return None

def build_cluster_label(center, terms, cid: int) -> str:
    idx = np.argsort(center)[::-1]
    tokens = []
    for i in idx:
        tok = terms[i]
        if len(tok) < 3 or tok in CLUSTER_STOPWORDS:
            continue
        if tok not in tokens:
            tokens.append(tok)
        if len(tokens) >= 2:
            break
    if tokens:
        return f"Other / {', '.join(tokens)}"
    return f"Other / Cluster {cid + 1}"

def main():
    if not ITEMS.exists():
        print("No items.csv found.")
        return

    it = pd.read_csv(ITEMS, low_memory=False)
    # name column
    name_col = (
        "product" if "product" in it.columns else
        ("product_norm" if "product_norm" in it.columns else
         ("product_raw" if "product_raw" in it.columns else None))
    )
    if not name_col:
        print("No product/product_norm column in items.csv")
        return

    it[name_col] = it[name_col].fillna("").astype(str)
    it["product_key"] = it[name_col].map(canonical_product_key)
    it = it[it["product_key"].str.len() > 0].copy()
    it["clean"] = it[name_col].map(lambda s: normalize_text(s, stopwords=None))
    it["line_total"] = pd.to_numeric(it.get("line_total"), errors="coerce").fillna(0)

    rule_labels = []
    for text in it["clean"]:
        tokens = set(text.split())
        label = rule_category(tokens)
        rule_labels.append(label)
    it["category_name"] = rule_labels
    it["category_source"] = np.where(it["category_name"].notna(), "rule", None)
    it["confidence"] = np.where(it["category_name"].notna(), 0.9, np.nan)

    uncat = it[it["category_name"].isna()].copy()
    cluster_labels = {}
    if not uncat.empty:
        N = uncat.shape[0]
        k = min(max(3, int(round(np.sqrt(N)))), 10)
        k = min(k, N) if N else 0
        if k >= 1:
            vect = TfidfVectorizer(min_df=1, ngram_range=(1, 2), stop_words=list(CLUSTER_STOPWORDS))
            X = vect.fit_transform(uncat["clean"])
            km = KMeans(n_clusters=k, n_init="auto", random_state=42)
            labs = km.fit_predict(X)
            terms = vect.get_feature_names_out()
            centers = km.cluster_centers_
            for cid in range(k):
                cluster_labels[cid] = build_cluster_label(centers[cid], terms, cid)
            uncat["category_name"] = [cluster_labels[c] for c in labs]
            uncat["category_source"] = "cluster"
            uncat["confidence"] = 0.5
            it.loc[uncat.index, ["category_name", "category_source", "confidence"]] = uncat[
                ["category_name", "category_source", "confidence"]
            ]

    display_name = (
        it.groupby("product_key")[name_col]
        .agg(lambda s: s.value_counts().index[0] if not s.empty else "")
        .rename("product")
    )
    mapping = (
        it.sort_values(["confidence", "line_total"], ascending=[False, False])
          .groupby("product_key", as_index=False)
          .first()
          .merge(display_name, on="product_key", how="left")
    )
    mapping = mapping[["product_key", "product", "category_name", "category_source", "confidence"]]
    mapping.to_csv(MAP_OUT, index=False)

    total_spend = float(it["line_total"].sum())
    breakdown = (
        it.groupby("category_name", as_index=False)
          .agg(spend=("line_total", "sum"), n_items=("category_name", "count"))
          .sort_values("spend", ascending=False)
    )
    breakdown["share"] = breakdown["spend"] / total_spend if total_spend else 0
    breakdown.to_csv(BREAK_OUT, index=False)

    meta = {
        "version": "rule-v1+cluster-v1",
        "n_items": int(it.shape[0]),
        "n_products": int(mapping.shape[0]),
        "rule_categories": RULE_ORDER,
        "cluster_k": int(len(cluster_labels)) if uncat is not None else 0,
        "cluster_labels": cluster_labels,
        "stopwords": sorted(CLUSTER_STOPWORDS),
    }
    META_OUT.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Wrote {MAP_OUT}, {BREAK_OUT}, {META_OUT}")

if __name__ == "__main__":
    main()
