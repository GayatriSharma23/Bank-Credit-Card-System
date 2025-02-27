import time
import json
import pandas as pd
import difflib
import os
from langchain_community.llms import Ollama
from tqdm import tqdm
import hashlib
import pickle

# Configuration
CONFIG = {
    'transaction_column': 'Narration',
    'model_name': "trnx_analyzer_mixtral:latest",
    'batch_size': 20,  # Reduced for better reliability
    'cache_file': 'transaction_cache.pkl',
    'categories': [
        "Food & Dining", "Groceries", "Shopping", "Transportation",
        "Utilities", "Housing", "Insurance", "Loans & EMIs", 
        "Entertainment", "Travel", "Healthcare", "Education",
        "Subscriptions", "Income", "Investments", "Cash Withdrawal",
        "Transfers", "Business Expenses", "Taxes", "Miscellaneous"
    ]
}

# Initialize cache
transaction_cache = {}
if os.path.exists(CONFIG['cache_file']):
    try:
        with open(CONFIG['cache_file'], 'rb') as f:
            transaction_cache = pickle.load(f)
        print(f"Loaded {len(transaction_cache)} cached transactions")
    except Exception as e:
        print(f"Error loading cache: {e}")

# Load CSV file
def load_data(filepath, limit=None):
    try:
        if limit:
            df = pd.read_csv(filepath, nrows=limit)
        else:
            df = pd.read_csv(filepath)
        print(f"CSV loaded successfully. {len(df)} rows found.")
        return df
    except Exception as e:
        print(f"Error loading CSV: {e}")
        raise

# Initialize Ollama LLM
def init_llm():
    try:
        return Ollama(model=CONFIG['model_name'])
    except Exception as e:
        print(f"Error initializing LLM: {e}")
        raise

# Generate batch indices
def create_batch_indices(data_length, batch_size):
    indices = list(range(0, data_length, batch_size))
    if indices[-1] != data_length:
        indices.append(data_length)
    return indices

# Create transaction fingerprint for caching
def create_fingerprint(transaction):
    return hashlib.md5(transaction.lower().encode()).hexdigest()

# Enhanced prompt with structured output requirement and examples
def create_categorization_prompt(transactions):
    categories_str = ", ".join(CONFIG['categories'])
    
    examples = [
        {"transaction": "ICICI BANK LTD EMI REPAYMENT", "category": "Loans & EMIs"},
        {"transaction": "RAZORPAY NETFLIX", "category": "Subscriptions"},
        {"transaction": "UPI/BIGBASKET/PAYMENT", "category": "Groceries"},
        {"transaction": "ATM/CASH WITHDRAWAL", "category": "Cash Withdrawal"},
        {"transaction": "SWIGGY ORDER PAYMENT", "category": "Food & Dining"}
    ]
    
    examples_str = "\n".join([f'"{e["transaction"]}" → {e["category"]}' for e in examples])
    
    prompt = f"""You are a financial transaction categorization expert.
Categorize each transaction into exactly ONE of these categories:
{categories_str}

Examples:
{examples_str}

For each transaction below, respond with the transaction text followed by " → " and then the category.
Return ONE category per transaction, with no explanation.

Transactions to categorize:
{chr(10).join([f'"{t}"' for t in transactions])}
"""
    return prompt

# Function to categorize transactions in batches with improved parsing
def categorize_transactions(transaction_names, llm, batch_num=0):
    # First check cache
    cached_transactions = []
    uncached_transactions = []
    transaction_map = {}  # To map back to original order
    
    for i, txn in enumerate(transaction_names):
        fingerprint = create_fingerprint(txn)
        if fingerprint in transaction_cache:
            cached_transactions.append((i, txn, transaction_cache[fingerprint]))
        else:
            uncached_transactions.append((i, txn))
            transaction_map[txn] = i
    
    print(f"Batch {batch_num}: {len(cached_transactions)} cached, {len(uncached_transactions)} to process")
    
    results = [None] * len(transaction_names)
    
    # Fill in cached results
    for i, txn, category in cached_transactions:
        results[i] = {
            'Transaction': txn,
            'Category': category,
            'Confidence': 1.0,  # Cached results get full confidence
            'Cached': True
        }
    
    # If we have uncached transactions to process
    if uncached_transactions:
        transactions_to_process = [t[1] for t in uncached_transactions]
        
        try:
            start_time = time.time()
            prompt = create_categorization_prompt(transactions_to_process)
            
            response = llm.invoke(prompt).strip()
            elapsed_time = time.time() - start_time
            print(f"Batch {batch_num} LLM processing completed in {elapsed_time:.2f} seconds")
            
            # Process response
            if not response:
                print(f"WARNING: Empty response from LLM for batch {batch_num}")
                for i, txn in uncached_transactions:
                    results[i] = {
                        'Transaction': txn,
                        'Category': 'Miscellaneous',  # Default for failed categorization
                        'Confidence': 0.0,
                        'Cached': False
                    }
            else:
                # Parse response and match to transactions
                parsed_results = {}
                
                for line in response.split('\n'):
                    line = line.strip()
                    if ' → ' in line or ' -> ' in line:
                        # Replace both arrow types
                        line = line.replace(' -> ', ' → ')
                        parts = line.split(' → ', 1)
                        
                        if len(parts) == 2:
                            txn, cat = parts
                            # Clean up the transaction text by removing quotes
                            txn = txn.strip('"\'').strip()
                            cat = cat.strip()
                            
                            # Validate category
                            if cat not in CONFIG['categories']:
                                cat = find_closest_category(cat, CONFIG['categories'])
                            
                            parsed_results[txn] = cat
                            
                            # Update cache
                            transaction_cache[create_fingerprint(txn)] = cat
                
                # Match results back to original transactions
                for txn, cat in parsed_results.items():
                    # Find closest matching transaction in our list
                    best_match = difflib.get_close_matches(txn, transaction_map.keys(), n=1, cutoff=0.7)
                    if best_match:
                        orig_txn = best_match[0]
                        i = transaction_map[orig_txn]
                        confidence = difflib.SequenceMatcher(None, txn, orig_txn).ratio()
                        results[i] = {
                            'Transaction': orig_txn,
                            'Category': cat,
                            'Confidence': confidence,
                            'Cached': False
                        }
                
                # Handle any transactions that weren't matched
                for i, txn in uncached_transactions:
                    if results[i] is None:
                        # Try fuzzy matching with parsed results
                        best_match = difflib.get_close_matches(txn, parsed_results.keys(), n=1, cutoff=0.6)
                        if best_match:
                            confidence = difflib.SequenceMatcher(None, txn, best_match[0]).ratio()
                            results[i] = {
                                'Transaction': txn,
                                'Category': parsed_results[best_match[0]],
                                'Confidence': confidence * 0.8,  # Reduce confidence for fuzzy matches
                                'Cached': False
                            }
                        else:
                            results[i] = {
                                'Transaction': txn,
                                'Category': 'Miscellaneous',
                                'Confidence': 0.0,
                                'Cached': False
                            }
        except Exception as e:
            print(f"Error processing batch {batch_num}: {e}")
            # Handle failure by assigning default category
            for i, txn in uncached_transactions:
                results[i] = {
                    'Transaction': txn,
                    'Category': 'Miscellaneous',
                    'Confidence': 0.0,
                    'Cached': False
                }
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    return results_df

# Find closest category if LLM returns non-standard category
def find_closest_category(category, valid_categories):
    match = difflib.get_close_matches(category, valid_categories, n=1, cutoff=0.6)
    return match[0] if match else "Miscellaneous"

# Main processing function
def process_transactions(df, batch_size=CONFIG['batch_size']):
    # Get unique transactions
    unique_transactions = df[CONFIG['transaction_column']].dropna().unique().tolist()
    print(f"Found {len(unique_transactions)} unique transactions to categorize")
    
    # Create batch indices
    indices = create_batch_indices(len(unique_transactions), batch_size)
    
    # DataFrame to store results
    categories_df = pd.DataFrame()
    
    # Initialize LLM
    llm = init_llm()
    
    # Process in batches with progress bar
    for i in tqdm(range(len(indices) - 1), desc="Processing batches"):
        batch_transactions = unique_transactions[indices[i]:indices[i+1]]
        if not batch_transactions:
            continue
            
        batch_results = categorize_transactions(batch_transactions, llm, i+1)
        categories_df = pd.concat([categories_df, batch_results], ignore_index=True)
        
        # Save cache periodically
        if i % 5 == 0 or i == len(indices) - 2:
            with open(CONFIG['cache_file'], 'wb') as f:
                pickle.dump(transaction_cache, f)
    
    return categories_df

# Apply categorization to full dataset
def apply_categories_to_dataset(df, categories_df):
    print("Applying categories to full dataset...")
    
    # Create a dictionary for faster lookups
    category_dict = dict(zip(categories_df['Transaction'], categories_df['Category']))
    confidence_dict = dict(zip(categories_df['Transaction'], categories_df['Confidence']))
    
    # Function to find best match with difflib
    def get_category(narration):
        if pd.isna(narration):
            return "Uncategorized", 0.0
            
        # Direct match
        if narration in category_dict:
            return category_dict[narration], confidence_dict[narration]
            
        # Fuzzy match
        match = difflib.get_close_matches(narration, list(category_dict.keys()), n=1, cutoff=0.7)
        if match:
            confidence = difflib.SequenceMatcher(None, narration, match[0]).ratio()
            return category_dict[match[0]], confidence
        return "Miscellaneous", 0.0
    
    # Apply function to get category and confidence
    df['Category'], df['Confidence'] = zip(*df[CONFIG['transaction_column']].apply(get_category))
    
    return df

# Generate insights from categorized data
def generate_insights(df):
    print("Generating insights...")
    
    # Ensure we have amount column
    if 'Debit' in df.columns:
        # Handle negative values properly
        df['Amount'] = df['Debit'].fillna(0) - df['Credit'].fillna(0)
    elif 'Amount' in df.columns:
        pass
    else:
        print("Warning: No amount column found for insights")
        df['Amount'] = 1  # Default for counting only
    
    # Summarize by category
    category_summary = df.groupby('Category').agg(
        TransactionCount=('Category', 'count'),
        TotalAmount=('Amount', 'sum'),
        AvgAmount=('Amount', 'mean'),
        MinAmount=('Amount', 'min'),
        MaxAmount=('Amount', 'max')
    ).sort_values('TotalAmount', ascending=False)
    
    # Calculate percentage of total
    total = category_summary['TotalAmount'].sum()
    category_summary['Percentage'] = (category_summary['TotalAmount'] / total * 100).round(2)
    
    # Time-based analysis if date column exists
    time_analysis = None
    date_columns = [col for col in df.columns if 'date' in col.lower() or 'day' in col.lower()]
    
    if date_columns:
        date_col = date_columns[0]
        try:
            df[date_col] = pd.to_datetime(df[date_col])
            df['Month'] = df[date_col].dt.to_period('M')
            time_analysis = df.groupby(['Month', 'Category']).agg(
                TotalAmount=('Amount', 'sum')
            ).reset_index()
        except:
            print(f"Could not convert {date_col} to datetime for time analysis")
    
    return {
        'category_summary': category_summary,
        'time_analysis': time_analysis,
        'confidence_stats': df['Confidence'].describe()
    }

# Main function
def main(filepath, output_file="categorized_transactions_enhanced.csv", limit=None):
    # Load data
    df = load_data(filepath, limit)
    
    # Process transactions
    categories_df = process_transactions(df)
    
    # Apply to full dataset
    df_categorized = apply_categories_to_dataset(df, categories_df)
    
    # Generate insights
    insights = generate_insights(df_categorized)
    
    # Save results
    df_categorized.to_csv(output_file, index=False)
    insights['category_summary'].to_csv("category_summary.csv")
    if insights['time_analysis'] is not None:
        insights['time_analysis'].to_csv("time_analysis.csv")
    
    print(f"\nDone! Output saved to {output_file}")
    print(f"Category summary saved to category_summary.csv")
    print(f"Average confidence: {insights['confidence_stats']['mean']:.2f}")
    
    return df_categorized, insights

if __name__ == "__main__":
    data_path = r'C:\Users\1137862\Desktop\Trnx_Analyser\AA_Data\IM_check_3_page_num_1.csv'
    main(data_path, limit=100)  # Set limit=None for full dataset
