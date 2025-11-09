# scripts/build_dashboard.py
import os, io, base64
import pandas as pd
import matplotlib.pyplot as plt

RCPT = "data/receipts.csv"
ITEM = "data/items.csv"
OUT_DIR = "artifacts"
OUT_HTML = os.path.join(OUT_DIR, "dashboard.html")

os.makedirs(OUT_DIR, exist_ok=True)

def _read_csv(path, **kw):
    if not os.path.exists(path):  # tolerate first runs
        return pd.DataFrame()
    return pd.read_csv(path, **kw)

# Load
receipts = _read_csv(RCPT)
items = _read_csv(ITEM)

# Parse dates & numbers safely
if not receipts.empty and "date" in receipts:
    receipts["date"] = pd.to_datetime(receipts["date"], errors="coerce")
else:
    receipts["date"] = pd.NaT

for col in ("total","subtotal","tax","confidence"):
    if col in receipts:
        receipts[col] = pd.to_numeric(receipts[col], errors="coerce")

if not items.empty:
    for col in ("line_total","qty","unit_price"):
        if col in items:
            items[col] = pd.to_numeric(items[col], errors="coerce")

# Basic stats
n_receipts = max(len(receipts) - 1, 0) if len(receipts) and receipts.columns[0] == receipts.iloc[0].get(receipts.columns[0], None) else len(receipts)  # defensive
n_receipts = receipts.shape[0]  # the line above is unnecessarily paranoid; use row count
n_items = items.shape[0]

total_spend = 0.0
if "total" in receipts and receipts["total"].notna().any():
    total_spend = float(receipts["total"].fillna(0).sum())
elif "line_total" in items:
    total_spend = float(items["line_total"].fillna(0).sum())

# Monthly series (by receipt date if available, else empty)
monthly_png_b64 = ""
if not receipts.empty and receipts["date"].notna().any():
    s = (receipts
         .dropna(subset=["date"])
         .set_index("date"))
    # Prefer receipt "total"; fallback to sum of items later if needed
    if "total" in s:
        m = s["total"].fillna(0).resample("MS").sum()
    else:
        m = pd.Series(dtype=float)
    if not m.empty:
        fig = plt.figure()
        m.plot(marker="o")
        plt.title("Monthly Spend (€)")
        plt.xlabel("Month")
        plt.ylabel("€")
        buf = io.BytesIO()
        plt.tight_layout()
        fig.savefig(buf, format="png", dpi=160)
        plt.close(fig)
        monthly_png_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

# Recent rows (lightweight preview)
def _safe_head(df, cols, n=10):
    cols = [c for c in cols if c in df.columns]
    if not cols or df.empty: return pd.DataFrame(columns=cols)
    return df[cols].tail(n).fillna("")

recent_receipts = _safe_head(receipts, ["date","vendor","total","currency","source_msg_id"], 10)
recent_items    = _safe_head(items,    ["receipt_id","line_no","product_raw","qty","unit_price","line_total"], 15)

# Render HTML (inline CSS, base64 img)
def _table(df):
    if df.empty: return "<em>No data</em>"
    return df.to_html(index=False, border=0, classes="tbl")

html = f"""<!doctype html>
<html>
<head>
<meta charset="utf-8"/>
<title>Naturalia – Dashboard</title>
<style>
 body{{font-family: -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, Helvetica, Arial, sans-serif; margin:24px;}}
 h1{{margin:0 0 8px 0}}
 .kpis{{display:flex; gap:16px; flex-wrap:wrap; margin:16px 0}}
 .kpi{{padding:12px 16px; border:1px solid #eee; border-radius:12px; box-shadow:0 1px 2px rgba(0,0,0,.04)}}
 .kpi .label{{font-size:12px; color:#666}}
 .kpi .value{{font-size:22px; font-weight:600}}
 .section{{margin-top:24px}}
 .tbl{{border-collapse:collapse; width:100%;}}
 .tbl th,.tbl td{{padding:8px 10px; border-bottom:1px solid #eee; font-size:14px; text-align:left}}
 .muted{{color:#666; font-size:12px}}
 .chart{{margin-top:8px;}}
</style>
</head>
<body>
  <h1>Naturalia – Dashboard</h1>
  <div class="kpis">
    <div class="kpi"><div class="label">Receipts</div><div class="value">{n_receipts}</div></div>
    <div class="kpi"><div class="label">Line items</div><div class="value">{n_items}</div></div>
    <div class="kpi"><div class="label">Total spend (€)</div><div class="value">{total_spend:,.2f}</div></div>
  </div>

  <div class="section">
    <h3>Monthly Spend</h3>
    {"<img class='chart' alt='Monthly Spend' src='data:image/png;base64," + monthly_png_b64 + "'/>" if monthly_png_b64 else "<span class='muted'>No dated receipts to chart yet.</span>"}
  </div>

  <div class="section">
    <h3>Recent Receipts</h3>
    {_table(recent_receipts)}
  </div>

  <div class="section">
    <h3>Recent Line Items</h3>
    {_table(recent_items)}
  </div>

  <div class="section muted">Generated from data/receipts.csv and data/items.csv.</div>
</body>
</html>
"""

with open(OUT_HTML, "w", encoding="utf-8") as f:
    f.write(html)

print("Wrote", OUT_HTML)

