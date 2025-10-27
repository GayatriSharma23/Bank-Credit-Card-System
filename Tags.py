# ------------------ CALCULATE TAG-LEVEL APE & % ------------------

# Initialize the tag columns with zeros to avoid NaN
for tag in ['13m', '25m', '37m']:
    persist_summary[f'ape_{tag}'] = 0.0

# Loop through each tag and aggregate APE within each distance bucket
for tag in ['13m', '25m', '37m']:
    tag_data = (
        df[df['persistency_tag'] == tag]
        .groupby('distance_bucket')['ape_lacs']
        .sum()
        .reindex(persist_summary['distance_bucket'])
        .fillna(0)
        .values
    )
    persist_summary[f'ape_{tag}'] = tag_data
    persist_summary[f'{tag}_pct'] = persist_summary.apply(
        lambda r: (r[f'ape_{tag}'] / r['total_ape'] * 100) if r['total_ape'] > 0 else 0,
        axis=1
    )
    persist_summary[f'{tag}_vs_company'] = persist_summary[f'{tag}_pct'] - COMPANY_AVG[tag]

display(
    persist_summary.style
    .background_gradient(cmap='YlGnBu')
    .set_caption("📊 Persistency by Distance Bucket (APE%)")
)
