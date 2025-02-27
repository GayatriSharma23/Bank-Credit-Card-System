import time
import pandas as pd
import difflib
from langchain_community.llms import Ollama
import os
import json

# Simple file-based cache using JSON (lighter than pickle)
CACHE_FILE = 'tx_categories_cache.json'
TRANSACTION_COLUMN = 'Narration'
BATCH_SIZE = 10  # Smaller batch size to prevent freezing
MODEL_NAME = "trnx_analyzer_mixtral:latest"

# Standard categories for consistency
CATEGORIES = [
    "Food", "Shopping", "Transport", "Utilities", "Housing", 
    "Insurance", "Loans", "Entertainment", "Travel", "Healthcare", 
    "Education", "Subscriptions", "Income", "Investments", "Cash", 
    "Transfers", "Business", "Miscellaneous"
]

# Load cache if it exists
def load_cache():
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, 'r') as f:
                return json.load(f)
        except:
            return {}
    return {}

# Save cache
def save_cache(cache):
    with open(CACHE_FILE, 'w') as f:
        json.dump(cache, f)

# Create smaller batches to prevent system freezing
def create_batches(items, batch_size):
    for i in range(0, len(items), batch_size):
        yield items[i:i+batch_size]

# Simple prompt with examples for better categorization
def create_prompt(transactions):
    categories_str = ", ".join(CATEGORIES)
    
    prompt = f"""Categorize these bank transactions into ONE of these categories:
{categories_str}

Examples:
"ICICI BANK EMI" → Loans
"NETFLIX PAYMENT" → Subscriptions
"SWIGGY" → Food
"ATM WITHDRAWAL" → Cash

Format: "Transaction" → Category

Transactions:
{chr(10).join([f'"{t}"' for t in transactions])}
"""
    return prompt

# Process transactions with proper error handling
def process_batch(transactions, llm, cache):
    # Check cache first
    to_process = []
    results = {}
    
    for tx in transactions:
        if tx in cache:
            results[tx] = cache[tx]
        else:
            to_process.append(tx)
    
    # Skip LLM call if all transactions are cached
    if not to_process:
        return results
    
    try:
        prompt = create_prompt(to_process)
        response = llm.invoke(prompt).strip()
        
        # Parse response
        for line in response.split('\n'):
            line = line.strip()
            if '→' in line:
                parts = line.split('→', 1)
                if len(parts) == 2:
                    tx, category = parts[0].strip(' "\''), parts[1].strip()
                    
                    # Validate category
                    if category not in CATEGORIES:
                        category = difflib.get_close_matches(category, CATEGORIES, n=1, cutoff=0.6)
                        category = category[0] if category else "Miscellaneous"
                    
                    # Find best match with original transaction
                    best_match = difflib.get_close_matches(tx, to_process, n=1, cutoff=0.8)
                    if best_match:
                        orig_tx = best_match[0]
                        results[orig_tx] = category
                        cache[orig_tx] = category
            
        # Handle any missing transactions
        for tx in to_process:
            if tx not in results:
                results[tx] = "Miscellaneous"
                cache[tx] = "Miscellaneous"
    
    except Exception as e:
        print(f"Error processing batch: {e}")
        # Fallback to Miscellaneous for errors
        for tx in to_process:
            if tx not in results:
                results[tx] = "Miscellaneous"
    
    return results

def main():
    # Load transaction data
    try:
        df = pd.read_csv(r'C:\Users\1137862\Desktop\Trnx_Analyser\AA_Data\IM_check_3_page_num_1.csv', nrows=100)
        print(f"CSV loaded successfully. {len(df)} rows found.")
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return
    
    # Load cache
    cache = load_cache()
    print(f"Loaded {len(cache)} cached transactions")
    
    # Get unique transactions
    unique_transactions = df[TRANSACTION_COLUMN].dropna().unique().tolist()
    print(f"Found {len(unique_transactions)} unique transactions")
    
    # Initialize LLM
    llm = Ollama(model=MODEL_NAME)
    
    # Process in small batches
    all_results = {}
    total_batches = (len(unique_transactions) + BATCH_SIZE - 1) // BATCH_SIZE
    
    for i, batch in enumerate(create_batches(unique_transactions, BATCH_SIZE)):
        print(f"Processing batch {i+1}/{total_batches} ({len(batch)} transactions)")
        start_time = time.time()
        
        batch_results = process_batch(batch, llm, cache)
        all_results.update(batch_results)
        
        elapsed = time.time() - start_time
        print(f"Batch completed in {elapsed:.2f} seconds")
        
        # Save cache every few batches to prevent data loss
        if i % 2 == 0:
            save_cache(cache)
    
    # Final cache save
    save_cache(cache)
    
    # Create results dataframe
    results_df = pd.DataFrame({
        'Transaction': list(all_results.keys()),
        'Category': list(all_results.values())
    })
    
    # Apply categories to original dataframe
    def get_category(tx):
        if pd.isna(tx):
            return "Unknown"
        
        if tx in all_results:
            return all_results[tx]
        
        # Try fuzzy matching
        match = difflib.get_close_matches(tx, list(all_results.keys()), n=1, cutoff=0.7)
        if match:
            return all_results[match[0]]
        return "Miscellaneous"
    
    df['Category'] = df[TRANSACTION_COLUMN].apply(get_category)
    
    # Save results
    output_file = "categorized_transactions_lightweight.csv"
    df.to_csv(output_file, index=False)
    
    # Generate a simple summary
    summary = df.groupby('Category').size().reset_index(name='Count')
    summary = summary.sort_values('Count', ascending=False)
    summary.to_csv("category_summary.csv", index=False)
    
    print(f"\nDone! Results saved to {output_file}")
    print("Top 5 categories:")
    print(summary.head(5))

if __name__ == "__main__":
    main()
