import re
import unicodedata

STOPWORDS = {
    "x",
    "kg",
    "g",
    "gr",
    "ml",
    "l",
    "pcs",
    "pc",
}

def normalize_text(s: str, stopwords=None) -> str:
    if s is None:
        return ""
    text = str(s).lower()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = re.sub(r"[^a-z0-9]+", " ", text)
    tokens = [t for t in text.split() if len(t) >= 2]
    if stopwords:
        tokens = [t for t in tokens if t not in stopwords]
    return " ".join(tokens).strip()

def canonical_product_key(name: str) -> str:
    return normalize_text(name, stopwords=STOPWORDS)
