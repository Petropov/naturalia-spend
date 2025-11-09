import pandas as pd, os
os.makedirs("artifacts", exist_ok=True)

rc = pd.read_csv("data/receipts.csv")
it = pd.read_csv("data/items.csv")

for c in ("total","subtotal","tax","confidence"):
    if c in rc: rc[c] = pd.to_numeric(rc[c], errors="coerce")
for c in ("line_total","qty","unit_price"):
    if c in it: it[c] = pd.to_numeric(it[c], errors="coerce")

sum_it = it.groupby("receipt_id", as_index=False)["line_total"].sum().rename(columns={"line_total":"items_sum"})
rep = rc.merge(sum_it, on="receipt_id", how="left")
rep["items_sum"] = rep["items_sum"].fillna(0.0)
rep["gap"] = rep["items_sum"] - rc["total"].fillna(0.0)
rep["gap_pct"] = (rep["gap"] / rc["total"].replace(0, pd.NA)) * 100
rep["status"] = "ok"
mask = rc["total"].fillna(0) > 0
rep.loc[mask & (rep["items_sum"] > rc["total"]*1.01), "status"] = "over"
rep.loc[mask & (rep["items_sum"] < rc["total"]*0.99), "status"] = "under"

rep.to_csv("artifacts/sanity_report.csv", index=False)

n = len(rep)
ok = int((rep["status"]=="ok").sum())
over = int((rep["status"]=="over").sum())
under = int((rep["status"]=="under").sum())
with open("artifacts/sanity_summary.txt","w",encoding="utf-8") as f:
    f.write(f"receipts: {n}, ok: {ok}, over: {over}, under: {under}\n")

print("Wrote artifacts/sanity_report.csv and artifacts/sanity_summary.txt")
