import pandas as pd

# Assuming df has:
# distance_bucket, policy_number (unique per policy),
# early_claim_1yr, early_claim_2yr, early_claim_3yr

# Step 1: Aggregate counts
summary = df.groupby('distance_bucket').agg(
    total_policies=('policy_number', 'count'),
    early_claim_1yr=('early_claim_1yr', 'sum'),
    early_claim_2yr=('early_claim_2yr', 'sum'),
    early_claim_3yr=('early_claim_3yr', 'sum')
).reset_index()

# Step 2: Calculate percentages wrt total policies
summary['early_claim_1yr_%'] = (summary['early_claim_1yr'] / summary['total_policies']) * 100
summary['early_claim_2yr_%'] = (summary['early_claim_2yr'] / summary['total_policies']) * 100
summary['early_claim_3yr_%'] = (summary['early_claim_3yr'] / summary['total_policies']) * 100

# Step 3: Round for neat presentation
summary = summary.round({
    'early_claim_1yr_%': 2,
    'early_claim_2yr_%': 2,
    'early_claim_3yr_%': 2
})

# Step 4: Optional — rename columns for presentation clarity
summary = summary.rename(columns={
    'distance_bucket': 'Distance Bucket',
    'total_policies': 'Total NOP',
    'early_claim_1yr': 'Early Claim (1 Year) NOP',
    'early_claim_1yr_%': 'Early Claim (1 Year) %',
    'early_claim_2yr': 'Early Claim (2 Year) NOP',
    'early_claim_2yr_%': 'Early Claim (2 Year) %',
    'early_claim_3yr': 'Early Claim (3 Year) NOP',
    'early_claim_3yr_%': 'Early Claim (3 Year) %'
})

print(summary)

