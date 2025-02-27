import pandas as pd
import numpy as np
import os
from config import CONFIG
from constants import CATEGORIES

def generate_summary(df):
    """
    Generate summary statistics and reports from categorized transactions
    
    Args:
        df (DataFrame): Categorized transaction DataFrame
        
    Returns:
        dict: Dictionary containing summary DataFrames
    """
    print("Generating summary reports...")
    
    # Create summary by category
    summary = df.groupby('Category').agg(
        Count=('Category', 'count'),
        Total_Amount=('Amount', 'sum'),
        Avg_Amount=('Amount', 'mean'),
        Min_Amount=('Amount', 'min'),
        Max_Amount=('Amount', 'max'),
        Avg_Confidence=('Confidence', 'mean')
    ).reset_index()
    
    # Sort by count descending
    summary = summary.sort_values('Count', ascending=False)
    
    # Format currency and percentage columns
    if 'Amount' in df.columns:
        summary['Total_Amount'] = summary['Total_Amount'].round(2)
        summary['Avg_Amount'] = summary['Avg_Amount'].round(2)
    
    summary['Avg_Confidence'] = summary['Avg_Confidence'].round(4)
    
    # Add percentage of total
    summary['Percentage'] = (summary['Count'] / summary['Count'].sum() * 100).round(2)
    
    # Get low confidence transactions
    low_confidence = df[df['Confidence'] < CONFIG['min_confidence']].copy()
    low_confidence = low_confidence.sort_values('Confidence')
    
    # Generate monthly trends if date information is available
    monthly_trends = None
    if 'Date' in df.columns:
        try:
            # Convert to datetime
            df['Date'] = pd.to_datetime(df['Date'])
            df['Month'] = df['Date'].dt.to_period('M')
            
            # Group by month and category
            monthly_trends = df.groupby(['Month', 'Category']).agg(
                Count=('Category', 'count'),
                Total_Amount=('Amount', 'sum')
            ).reset_index()
            
            # Save monthly trends
            monthly_trends.to_csv("monthly_category_trends.csv", index=False)
        except Exception as e:
            print(f"Error generating monthly trends: {e}")
    
    return {
        'summary': summary,
        'low_confidence': low_confidence,
        'monthly_trends': monthly_trends
    }

def analyze_merchant_patterns(df):
    """
    Analyze merchant patterns to improve categorization
    
    Args:
        df (DataFrame): Categorized transaction DataFrame
        
    Returns:
        dict: Dictionary of dominant merchant patterns
    """
    merchant_patterns = {}
    
    # Extract merchant name from each transaction
    for tx, category in zip(df[CONFIG['transaction_column']], df['Category']):
        if pd.isna(tx) or not tx:
            continue
            
        # Extract potential merchant name (first 1-2 words)
        words = tx.split()
        if not words:
            continue
            
        merchant = words[0].lower()
        if len(words) > 1:
            merchant = f"{merchant} {words[1].lower()}"
        
        if merchant not in merchant_patterns:
            merchant_patterns[merchant] = {}
            
        if category not in merchant_patterns[merchant]:
            merchant_patterns[merchant][category] = 0
            
        merchant_patterns[merchant][category] += 1
    
    # Find dominant categories for each merchant
    dominant_merchants = {}
    for merchant, categories in merchant_patterns.items():
        if len(categories) < 2:
            dominant_cat = next(iter(categories.keys()))
            dominant_merchants[merchant] = dominant_cat
            continue
            
        total = sum(categories.values())
        max_cat = max(categories.items(), key=lambda x: x[1])
        if max_cat[1] / total > 0.8:  # 80% threshold
            dominant_merchants[merchant] = max_cat[0]
    
    # Save merchant patterns for analysis
    if len(dominant_merchants) > 0:
        with open('detected_merchant_patterns.txt', 'w') as f:
            for merchant, category in sorted(dominant_merchants.items()):
                f.write(f"{merchant}: {category}\n")
    
    print(f"Analyzed {len(merchant_patterns)} merchant patterns")
    print(f"Found {len(dominant_merchants)} dominant merchant patterns")
    
    return dominant_merchants

def generate_category_insights(df):
    """
    Generate insights about spending patterns and anomalies
    
    Args:
        df (DataFrame): Categorized transaction DataFrame
        
    Returns:
        DataFrame: Insights dataframe
    """
    insights = []
    
    # Skip if no Amount column
    if 'Amount' not in df.columns:
        return pd.DataFrame(insights)
    
    try:
        # Find outliers in each category (transactions > 2 std devs from mean)
        for category in CATEGORIES:
            cat_df = df[df['Category'] == category]
            if len(cat_df) < 5:  # Need enough data for meaningful stats
                continue
                
            mean = cat_df['Amount'].mean()
            std = cat_df['Amount'].std()
            threshold = mean + 2 * std
            
            outliers = cat_df[cat_df['Amount'] > threshold]
            
            if len(outliers) > 0:
                for _, row in outliers.iterrows():
                    insights.append({
                        'Type': 'Outlier',
                        'Category': category,
                        'Transaction': row[CONFIG['transaction_column']],
                        'Amount': row['Amount'],
                        'Date': row.get('Date', 'N/A'),
                        'Description': f"Unusually high amount ({row['Amount']:.2f} vs avg {mean:.2f})"
                    })
        
        # Check for sudden increases in category spending
        if 'Date' in df.columns and 'Month' in df.columns:
            monthly_spending = df.groupby(['Month', 'Category'])['Amount'].sum().reset_index()
            
            for category in CATEGORIES:
                cat_monthly = monthly_spending[monthly_spending['Category'] == category]
                if len(cat_monthly) < 3:  # Need at least 3 months of data
                    continue
                    
                cat_monthly = cat_monthly.sort_values('Month')
                cat_monthly['Change'] = cat_monthly['Amount'].pct_change() * 100
                
                spikes = cat_monthly[cat_monthly['Change'] > 50]  # 50% increase
                
                if len(spikes) > 0:
                    for _, row in spikes.iterrows():
                        insights.append({
                            'Type': 'Trend',
                            'Category': category,
                            'Transaction': 'Multiple',
                            'Amount': row['Amount'],
                            'Date': row['Month'],
                            'Description': f"Spending increased by {row['Change']:.1f}% from previous month"
                        })
    
    except Exception as e:
        print(f"Error generating insights: {e}")
    
    # Convert to DataFrame
    insights_df = pd.DataFrame(insights)
    
    # Save insights
    if len(insights_df) > 0:
        insights_df.to_csv("transaction_insights.csv", index=False)
        print(f"Generated {len(insights_df)} spending insights")
    
    return insights_df
