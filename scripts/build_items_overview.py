from pathlib import Path
import pandas as pd

DATA = Path("data")
OUT  = Path("reports/items_overview.html")
OUT.parent.mkdir(exist_ok=True)

items = pd.read_csv(DATA/"items.csv")
receipts = pd.read_csv(DATA/"receipts.csv", dtype=str)
receipts["date"] = pd.to_datetime(receipts["date"], errors="coerce", utc=True).dt.tz_localize(None)

df = items.merge(receipts[["receipt_id","date","store","postcode"]], on="receipt_id", how="left")
df = df.dropna(subset=["line_total"]).copy()
df["line_total"] = df["line_total"].astype(float)
df["qty"] = df["qty"].astype(int)
df["month"] = df["date"].dt.to_period("M").dt.to_timestamp()
df["store"] = df["store"].fillna("")

def money(x): return "€{:,.2f}".format(x).replace(",", " ")

top_products = (df.groupby("product", as_index=False)["line_total"].sum()
                  .sort_values("line_total", ascending=False))
by_month = (df.groupby(["month"], as_index=False)["line_total"].sum()
              .sort_values("month"))
by_store = (df.groupby(["store"], as_index=False)["line_total"].sum()
              .sort_values("line_total", ascending=False))

history = (df.sort_values(["date","product"])
             [["date","product","qty","unit_price","line_total","store"]].copy())
history["date"] = history["date"].dt.strftime("%Y-%m-%d")

for d in (top_products, by_month, by_store, history):
    if "line_total" in d: d["line_total"] = d["line_total"].map(money)
    if "unit_price" in d: d["unit_price"] = d["unit_price"].map(money)

tp_tbl  = top_products.rename(columns={"line_total":"Spend (€)"})
bm_tbl  = by_month.rename(columns={"line_total":"Spend (€)", "month":"Month"})
bs_tbl  = by_store.rename(columns={"line_total":"Spend (€)", "store":"Store"})
hist_tbl= history.rename(columns={"line_total":"Spend (€)", "unit_price":"Unit (€)"})

def table(df, title):
    if df.empty: return f"<h2>{title}</h2><p class='muted'>No data.</p>"
    return f"<h2>{title}</h2>" + df.to_html(index=False, escape=False)

html = f"""<!doctype html>
<meta charset="utf-8">
<title>Naturalia — Items Overview</title>
<style>
body{{font-family:-apple-system,BlinkMacSystemFont,Segoe UI,Inter,Roboto,Helvetica,Arial,sans-serif;margin:24px}}
h1{{margin:0 0 8px}} h2{{margin:18px 0 10px}} .muted{{color:#666}}
table{{border-collapse:collapse;width:100%;background:#fff;margin:8px 0 16px}}
th,td{{border:1px solid #e7e7e7;padding:8px 10px;text-align:left}}
th{{background:#fafafa}}
</style>
<h1>Naturalia — Items Overview</h1>
<div class="muted">Built from receipts.csv + items.csv</div>
{table(tp_tbl, "Top products by spend")}
{table(bm_tbl, "Spend by month")}
{table(bs_tbl, "Spend by store")}
{table(hist_tbl, "Purchase history")}
"""
OUT.write_text(html, encoding="utf-8")
print("Wrote", OUT)
