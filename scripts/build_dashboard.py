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
