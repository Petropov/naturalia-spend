import pandas as pd, matplotlib.pyplot as plt
from pathlib import Path

RCPT='data/receipts.csv'
ITEM='data/items.csv'
OUT_MD='reports/summary.md'

def main():
    Path('reports').mkdir(exist_ok=True)
    if not Path(RCPT).exists():
        Path(OUT_MD).write_text("# Naturalia Report\nNo data yet.\n"); return
    df = pd.read_csv(RCPT, parse_dates=['date'])
    if df.empty or df['date'].isna().all():
        Path(OUT_MD).write_text("# Naturalia Report\nNo data yet.\n"); return

    df = df.dropna(subset=['date'])
    df['week'] = df['date'].dt.to_period('W').dt.start_time
    weekly = df.groupby('week', as_index=False)['total'].sum(numeric_only=True).sort_values('week')
    weekly['wow_change'] = weekly['total'].pct_change().fillna(0)
    weekly['t4w'] = weekly['total'].rolling(4).mean()
    weekly['t12w'] = weekly['total'].rolling(12).mean()

    # chart
    plt.figure(figsize=(8,4))
    plt.plot(weekly['week'], weekly['total'])
    plt.plot(weekly['week'], weekly['t4w'])
    plt.plot(weekly['week'], weekly['t12w'])
    plt.title('Weekly Spend (Total / T4W / T12W)')
    plt.xlabel('Week'); plt.ylabel('EUR'); plt.tight_layout()
    plt.savefig('reports/weekly.png'); plt.close()

    lifetime_sum = float(df['total'].sum())
    wow_pct = float(weekly.iloc[-1]['wow_change']*100) if not weekly.empty else 0.0

    # price changes
    price_changes_md = "No item-level price data."
    if Path(ITEM).exists():
        it = pd.read_csv(ITEM)
        if not it.empty:
            it['eff_unit'] = it.apply(
                lambda r: (r['line_total']/r['qty']) if (pd.notna(r['qty']) and str(r['qty']) not in ('', '0') and pd.notna(r['line_total'])) else r['unit_price'],
                axis=1
            )
            it['eff_unit'] = pd.to_numeric(it['eff_unit'], errors='coerce')
            it = it.merge(df[['receipt_id','week']], on='receipt_id', how='left')
            grp = it.dropna(subset=['product_norm','eff_unit','week']) \
                   .groupby(['product_norm','week'])['eff_unit'] \
                   .mean().reset_index().sort_values(['product_norm','week'])
            last_two = grp.groupby('product_norm').tail(2)
            deltas = []
            for p, g in last_two.groupby('product_norm'):
                if len(g)==2:
                    g=g.sort_values('week')
                    prev, cur = g.iloc[0]['eff_unit'], g.iloc[1]['eff_unit']
                    if pd.notna(prev) and pd.notna(cur) and prev>0:
                        deltas.append((p, prev, cur, (cur-prev)/prev*100))
            deltas = sorted(deltas, key=lambda x: abs(x[3]), reverse=True)[:15]
            if deltas:
                names = [d[0][:24] for d in deltas]
                pct = [d[3] for d in deltas]
                plt.figure(figsize=(8,4))
                plt.barh(names, pct)
                plt.title('Top Product Price Changes (%)')
                plt.tight_layout()
                plt.savefig('reports/price_changes.png'); plt.close()
                lines = ["| Product | Prev | Curr | Δ% |","|---|---:|---:|---:|"]
                for n,pr,cu,pctg in deltas:
                    lines.append(f"| {n} | {pr:.2f} | {cu:.2f} | {pctg:+.1f}% |")
                price_changes_md = "\n".join(lines)

    md = []
    md.append("# Naturalia Spend Report")
    md.append(f"- **Lifetime spend:** €{lifetime_sum:.2f}")
    md.append(f"- **Latest WoW change:** {wow_pct:+.1f}%")
    md.append("\n## Weekly trend\n![weekly](weekly.png)\n")
    md.append("## Price changes (same products)\n")
    md.append(price_changes_md)
    Path(OUT_MD).write_text("\n".join(md))

if __name__ == '__main__':
    main()
