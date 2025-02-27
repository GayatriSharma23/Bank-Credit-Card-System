import pandas as pd
import os
import sys
import time
from config import CONFIG
from data_handlers import load_cache, save_cache
from processors import process_transactions, apply_categories
from analyzers import generate_summary, analyze_merchant_patterns, generate_category_insights
from feedback import apply_feedback_to_results
from constants import MERCHANT_CATEGORIES

def main(input_file=None, limit_rows=None):
    """
    Main function to run the transaction categorization process
    
    Args:
        input_file (str): Path to input CSV file. If None, will use configured default.
        limit_rows (int): Limit processing to this many rows. If None, process all rows.
        
    Returns:
        bool: True if processing was successful, False otherwise
    """
    print("=" * 60)
    print(" Transaction Categorization System ")
    print("=" * 60)
    print(f"Using model: {CONFIG['model_name']}")
    
    # Handle input file
    if input_file is None:
        if len(sys.argv) > 1:
            input_file = sys.argv[1]
        else:
            input_file = input("Enter path to CSV file: ").strip()
            if not input_file:
                print("No input file provided. Exiting.")
                return False
    
    # Load data
    start_time = time.time()
    try:
        print(f"Loading data from: {input_file}")
        if limit_rows:
            df = pd.read_csv(input_file, nrows=limit_rows)
            print(f"CSV loaded successfully. Processing {limit_rows} rows.")
        else:
            df = pd.read_csv(input_file)
            print(f"CSV loaded successfully. {len(df)} rows found.")
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return False
    
    # Check for required columns
    if CONFIG['transaction_column'] not in df.columns:
        print(f"Error: Required column '{CONFIG['transaction_column']}' not found in CSV.")
        print(f"Available columns: {', '.join(df.columns)}")
        return False
    
    # Process transactions
    print(f"Processing transactions in batches of {CONFIG['batch_size']}...")
    results_df, category_dict, confidence_dict = process_transactions(df)
    
    # Apply categories to full dataset
    print("Applying categories to dataset...")
    df_categorized = apply_categories(df, category_dict, confidence_dict)
    
    # Apply any existing user feedback
    print("Applying user feedback...")
    df_categorized = apply_feedback_to_results(df_categorized)
    
    # Analyze merchant patterns for continuous learning
    print("Analyzing merchant patterns...")
    new_merchants = analyze_merchant_patterns(df_categorized)
    
    # Generate summary
    summary = generate_summary(df_categorized)
    
    # Generate insights
    insights = generate_category_insights(df_categorized)
    
    # Create output directory if it doesn't exist
    output_dir = "output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Save results
    output_file = os.path.join(output_dir, CONFIG['output_file'])
    df_categorized.to_csv(output_file, index=False)
    
    summary_file = os.path.join(output_dir, "category_summary.csv")
    summary['summary'].to_csv(summary_file)
    
    low_conf_file = os.path.join(output_dir, "low_confidence_transactions.csv")
    summary['low_confidence'].to_csv(low_conf_file)
    
    # Save monthly trends if available
    if summary['monthly_trends'] is not None:
        monthly_file = os.path.join(output_dir, "monthly_category_trends.csv")
        summary['monthly_trends'].to_csv(monthly_file)
    
    # Calculate processing time
    elapsed_time = time.time() - start_time
    
    # Print statistics
    print("\n" + "=" * 60)
    print(" Results Summary ")
    print("=" * 60)
    print(f"Total transactions processed: {len(df_categorized)}")
    print(f"Total categories identified: {len(summary['summary'])}")
    print(f"Average confidence: {df_categorized['Confidence'].mean():.2f}")
    print(f"Low confidence transactions: {len(summary['low_confidence'])} ({len(summary['low_confidence'])/len(df_categorized)*100:.1f}%)")
    print(f"New merchant patterns identified: {len(new_merchants)}")
    print(f"Processing time: {elapsed_time:.2f} seconds")
    
    # Print top categories
    print("\nTop Categories:")
    top_cats = summary['summary'].head(5)
    for _, row in top_cats.iterrows():
        print(f"  {row['Category']}: {row['Count']} transactions ({row['Percentage']}%)")
    
    # Print insights
    if len(insights) > 0:
        print(f"\nGenerated {len(insights)} insights. See transaction_insights.csv for details.")
    
    print("\n" + "=" * 60)
    print(f"Done! Results saved to {output_dir}/ directory")
    print("=" * 60)
    
    return True

if __name__ == "__main__":
    main()
