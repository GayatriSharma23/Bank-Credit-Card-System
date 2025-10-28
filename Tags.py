import pandas as pd
import numpy as np

# Define company averages (for comparison)
COMPANY_AVG = {"13m": 76.6, "25m": 66.0, "37m": 56.0}

# Define persistency tags to evaluate
tags = ['13m', '25m', '37m']

# Prepare summary list
persistency_summary = []

for tag in tags:
    tag_col = f"{tag.replace('m', '')}c_{tag.replace('m', '')}nc_tag"  # e.g. 13c_13nc_tag
    if tag_col not in df.columns:
        print(f"⚠️ Missing column: {tag_col}")
        continue

    # Filter rows that actually belong to this persistency period
    df_tag = df[df['persistency_tag'].str.lower() == tag.lower()].copy()

    # Clean column
    df_tag[tag_col] = df_tag[tag_col].astype(str).str.upper().str.strip()

    # Calculate APEs
    continued_ape = df_tag.loc[df_tag[tag_col] == f"{tag.replace('m','').upper()}C", 'ape_lacs'].sum()
    total_ape = df_tag['ape_lacs'].sum()

    # Group by distance bucket
    bucket_summary = (
        df_tag.groupby('distance_bucket', dropna=False)
        .agg(
            total_ape=('ape_lacs', 'sum'),
            continued_ape=(lambda x: df_tag.loc[
                (df_tag['distance_bucket'] == x.name) &
                (df_tag[tag_col] == f"{tag.replace('m','').upper()}C"), 'ape_lacs'
            ].sum())
        )
        .reset_index()
    )

    # Calculate Persistency %
    bucket_summary[f'{tag}_persistency_pct'] = (
        (bucket_summary['continued_ape'] / bucket_summary['total_ape']) * 100
    ).fillna(0)

    # Compare with company average
    company_avg = COMPANY_AVG.get(tag, np.nan)
    bucket_summary[f'{tag}_variance_vs_avg'] = bucket_summary[f'{tag}_persistency_pct'] - company_avg

    persistency_summary.append(bucket_summary)

# Combine all persistency tables
persistency_summary_df = persistency_summary[0]
for temp_df in persistency_summary[1:]:
    persistency_summary_df = persistency_summary_df.merge(temp_df, on='distance_bucket', how='outer')

display(persistency_summary_df.style.background_gradient(cmap='YlGnBu').set_caption("📊 Persistency Summary by Distance Bucket"))

