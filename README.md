# Naturalia Spend Tracker

**Automated expense intelligence** for Naturalia receipts: This project ingests receipts directly from Gmail, parses PDFs, and builds an analytical dashboard — fully automated, reproducible, and improving as more data arrives.

I was just curious to understand the patterns of what i buy in naturalia, and to figure out how prices increase.

---

## Architecture Overview

### 1. Ingestion (Gmail → PDF → CSV)
- **Source:** Gmail attachments matching  
  `from:naturalia has:attachment filename:pdf newer_than:<days>d`
- **Script:** `src/gmail_pull.py`
- **Secrets (GitHub Actions):**
  - `GMAIL_CREDENTIALS_B64`
  - `GMAIL_TOKEN_B64`
- **Outputs:**
  - `data/receipts.csv`
  - `data/items.csv`
  - `data/hashes.txt` (SHA256 of each PDF for deduplication)

### 2. Parsing (PDF → structured text)
- **Module:** `src/parse_receipt.py`
- **Libraries:** `pdfminer.six` + `pytesseract`
- Extracts:
  - Receipt metadata: date, store, total, currency
  - Line items: product, qty, unit price, line total
- Handles both digital and scanned PDFs.

### 3. Pipeline Orchestration
- **Workflow:** `.github/workflows/report.yml`
- **Entrypoint:** `scripts/build_dashboard.py`
- Executes ingestion, parsing, unsupervised category learning, and dashboard generation.
- **Outputs:**
  - `artifacts/dashboard.html`
  - `artifacts/categories_*.csv`
  - `artifacts/categories_meta.json`

---

## Dashboard (`artifacts/dashboard.html`)

The dashboard provides a receipt-based view of spend evolution, product behavior, and learned categories.

### Sections
1. **Reconciliation (Items vs Receipts)**  
   Verifies that all breakdowns sum to the total amount spent.
2. **Spend by Month / ISO Week / Type of Day / Weekday**  
   Temporal breakdown of total spend, consistent across slices.
3. **Price Change (Last vs Previous)**  
   Tracks unit-price evolution for repeated products.
4. **Spend by Category (Learned)**  
   Data-driven product grouping via the unsupervised learner.
5. **Price Percentiles (Unit Price)**  
   P10–P90 statistics for item prices.

---

## Machine-Learned Categories (Unsupervised, French-aware)

### Script: `scripts/learn_categories.py`

**Goal:** Automatically group products by meaning, not by manual rules.

- **Text normalization:**  
  Lowercased, cleans accents and packaging info (e.g., removes “20CL”, “1KG”).
- **Stopwords:**  
  Built-in French stopword list (`de, du, des, le, la, les, à, pour, avec, sans, ...`).
- **Vectorization:**  
  TF-IDF (1- and 2-grams) trained and persisted (`artifacts/tfidf.pkl`).
- **Model:**  
  KMeans with **consensus K** selection via silhouette + Davies–Bouldin scores.
- **Spend-weighted labeling:**  
  Category names derived from top TF-IDF terms weighted by cluster spend.
- **Persistence:**  
  - `tfidf.pkl` → stable vocabulary  
  - `kmeans.pkl` → stable cluster IDs across runs

**Outputs**
| File | Description |
|------|--------------|
| `artifacts/categories_learned.csv` | Product → category |
| `artifacts/categories_breakdown.csv` | Category → spend |
| `artifacts/categories_meta.json` | Cluster diagnostics (k, metrics, label map) |

As new receipts arrive, the learner retrains on the full dataset, improving category boundaries automatically — no hand-coded taxonomy required.

---

## Data Model

| File | Purpose | Key Columns |
|------|----------|--------------|
| `data/receipts.csv` | One row per receipt | `receipt_id, date, store, total, currency, msg_id` |
| `data/items.csv` | One row per item | `receipt_id, line_no, product, qty, unit_price, line_total, currency` |
| `artifacts/categories_learned.csv` | Learned mapping | `product, category_name` |
| `artifacts/categories_breakdown.csv` | Spend summary | `category_name, spend` |

---

## Developer Guide

### Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install joblib scikit-learn pdfminer.six pytesseract



===========================================

### 🧭 from a completely blank terminal

#### 1️⃣ navigate into your repo

```bash
cd ~/naturalia-spend
```

#### 2️⃣ activate your virtual environment

(macOS / Linux)

```bash
source .venv/bin/activate
```

(Windows PowerShell)

```powershell
.\.venv\Scripts\Activate.ps1
```

if the venv doesn’t exist (e.g. new machine):

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install joblib scikit-learn pdfminer.six pytesseract
```

#### 3️⃣ rebuild and open the dashboard

```bash
python scripts/build_dashboard.py && open artifacts/dashboard.html
```

---

### 💡 optional shortcuts

* **just rebuild, no browser:**

  ```bash
  python scripts/build_dashboard.py
  ```
* **rerun the full Gmail ingestion + report via GitHub Actions:**

  ```bash
  gh workflow run report.yml --ref main -f lookback_days=90
  ```

---

### 🪄 TL;DR single command (once venv is set up)

```bash
cd ~/naturalia-spend && source .venv/bin/activate && python scripts/build_dashboard.py && open artifacts/dashboard.html
```

That’s all you’ll ever need in a blank shell to regenerate the latest Naturalia spend dashboard.

