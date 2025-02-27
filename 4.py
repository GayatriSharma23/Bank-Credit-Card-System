import time
import json
import pandas as pd
import difflib
import re
import os
from langchain_community.llms import Ollama

# Configuration
CONFIG = {
    'transaction_column': 'Narration',
    'model_name': "trnx_analyzer_mixtral:latest",
    'batch_size': 15,
    'cache_file': 'tx_categories_enhanced_cache.json',
    'min_confidence': 0.7,
    'output_file': "categorized_transactions_enhanced.csv"
}

# Standard categories
CATEGORIES = [
    "Food & Dining", "Groceries", "Shopping", "Transportation",
    "Utilities", "Housing", "Insurance", "Loans & EMIs", 
    "Entertainment", "Travel", "Healthcare", "Education",
    "Subscriptions", "Income", "Investments", "Cash Withdrawal",
    "Transfers", "Business Expenses", "Miscellaneous"
]

# Merchant to category mapping dictionary
MERCHANT_CATEGORIES = {
    # Food & Dining
    "zomato": "Food & Dining",
    "swiggy": "Food & Dining",
    "uber eats": "Food & Dining",
    "restaurant": "Food & Dining",
    "food": "Food & Dining",
    "dominoes": "Food & Dining",
    "pizza": "Food & Dining",
    "mcdonald": "Food & Dining",
    "kfc": "Food & Dining",
    "cafe": "Food & Dining",
    "burger": "Food & Dining",
    
    # Groceries
    "bigbasket": "Groceries",
    "grofers": "Groceries",
    "grocery": "Groceries",
    "supermarket": "Groceries",
    "market": "Groceries",
    "fruit": "Groceries",
    "vegetable": "Groceries",
    "wholesale": "Groceries",
    
    # Shopping
    "amazon": "Shopping",
    "flipkart": "Shopping",
    "myntra": "Shopping",
    "ajio": "Shopping",
    "retail": "Shopping",
    "store": "Shopping",
    "mall": "Shopping",
    "shop": "Shopping",
    "clothing": "Shopping",
    "fashion": "Shopping",
    
    # Transportation
    "uber": "Transportation",
    "ola": "Transportation",
    "cab": "Transportation",
    "taxi": "Transportation",
    "metro": "Transportation",
    "train": "Transportation",
    "railway": "Transportation",
    "irctc": "Transportation",
    "bus": "Transportation",
    "transport": "Transportation",
    "fuel": "Transportation",
    "petrol": "Transportation",
    "diesel": "Transportation",
    
    # Utilities
    "electricity": "Utilities",
    "power": "Utilities",
    "water": "Utilities",
    "gas": "Utilities",
    "broadband": "Utilities",
    "internet": "Utilities",
    "wifi": "Utilities",
    "utility": "Utilities",
    "bill": "Utilities",
    
    # Housing
    "rent": "Housing",
    "house": "Housing",
    "apartment": "Housing",
    "flat": "Housing",
    "maintenance": "Housing",
    "property": "Housing",
    "real estate": "Housing",
    
    # Insurance
    "insurance": "Insurance",
    "policy": "Insurance",
    "premium": "Insurance",
    "life insurance": "Insurance",
    "health insurance": "Insurance",
    "motor insurance": "Insurance",
    "vehicle insurance": "Insurance",
    
    # Loans & EMIs
    "loan": "Loans & EMIs",
    "emi": "Loans & EMIs",
    "repayment": "Loans & EMIs",
    "interest": "Loans & EMIs",
    "principal": "Loans & EMIs",
    "credit card": "Loans & EMIs",
    "credit": "Loans & EMIs",
    "homeloan": "Loans & EMIs",
    "carloan": "Loans & EMIs",
    "personalloan": "Loans & EMIs",
    "mortgage": "Loans & EMIs",
    "finance": "Loans & EMIs",
    
    # Entertainment
    "movie": "Entertainment",
    "cinema": "Entertainment",
    "theater": "Entertainment",
    "netflix": "Entertainment",
    "amazon prime": "Entertainment",
    "hotstar": "Entertainment",
    "disney": "Entertainment",
    "bookmyshow": "Entertainment",
    "music": "Entertainment",
    "spotify": "Entertainment",
    "event": "Entertainment",
    "concert": "Entertainment",
    "game": "Entertainment",
    "gaming": "Entertainment",
    
    # Travel
    "hotel": "Travel",
    "resort": "Travel",
    "booking": "Travel",
    "flight": "Travel",
    "airline": "Travel",
    "travel": "Travel",
    "tour": "Travel",
    "vacation": "Travel",
    "holiday": "Travel",
    "trip": "Travel",
    "makemytrip": "Travel",
    "goibibo": "Travel",
    "oyo": "Travel",
    
    # Healthcare
    "hospital": "Healthcare",
    "doctor": "Healthcare",
    "clinic": "Healthcare",
    "medical": "Healthcare",
    "medicine": "Healthcare",
    "pharmacy": "Healthcare",
    "health": "Healthcare",
    "dental": "Healthcare",
    "eye": "Healthcare",
    "optical": "Healthcare",
    "diagnostic": "Healthcare",
    "pathology": "Healthcare",
    "lab": "Healthcare",
    
    # Education
    "school": "Education",
    "college": "Education",
    "university": "Education",
    "tuition": "Education",
    "course": "Education",
    "class": "Education",
    "education": "Education",
    "learning": "Education",
    "books": "Education",
    "stationery": "Education",
    "coaching": "Education",
    
    # Subscriptions
    "subscription": "Subscriptions",
    "membership": "Subscriptions",
    "prime": "Subscriptions",
    "premium": "Subscriptions",
    
    # Income
    "salary": "Income",
    "income": "Income",
    "dividend": "Income",
    "interest": "Income",
    "bonus": "Income",
    "commission": "Income",
    "payment received": "Income",
    "refund": "Income",
    "reimbursement": "Income",
    
    # Investments
    "investment": "Investments",
    "stock": "Investments",
    "share": "Investments",
    "mutual fund": "Investments",
    "fd": "Investments",
    "fixed deposit": "Investments",
    "bond": "Investments",
    "demat": "Investments",
    "zerodha": "Investments",
    "groww": "Investments",
    "etf": "Investments",
    "sip": "Investments",
    
    # Cash Withdrawal
    "atm": "Cash Withdrawal",
    "withdrawal": "Cash Withdrawal",
    "cash withdrawal": "Cash Withdrawal",
    "self withdrawal": "Cash Withdrawal",
    
    # Transfers
    "transfer": "Transfers",
    "upi": "Transfers",
    "neft": "Transfers",
    "rtgs": "Transfers",
    "imps": "Transfers",
    "remittance": "Transfers",
    "paytm": "Transfers",
    "gpay": "Transfers",
    "phonepe": "Transfers",
    "bhim": "Transfers",
    
    # Business Expenses
    "business": "Business Expenses",
    "office": "Business Expenses",
    "client": "Business Expenses",
    "vendor": "Business Expenses",
    "supplier": "Business Expenses",
    "material": "Business Expenses",
    "equipment": "Business Expenses",
    "machinery": "Business Expenses"
}

# Load cache
def load_cache():
    if os.path.exists(CONFIG['cache_file']):
        try:
            with open(CONFIG['cache_file'], 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading cache: {e}")
            return {}
    return {}

# Save cache
def save_cache(cache):
    try:
        with open(CONFIG['cache_file'], 'w') as f:
            json.dump(cache, f)
    except Exception as e:
        print(f"Error saving cache: {e}")

# Normalize transaction text for better matching
def normalize_transaction(transaction):
    """Clean and standardize transaction descriptions"""
    # Handle NaN values
    if pd.isna(transaction):
        return ""
        
    # Convert to lowercase
    tx = transaction.lower()
    
    # Remove common filler words
    fillers = ["ltd", "limited", "pvt", "private", "transaction", "txn", "ref", "reference", "payment to", "payment for"]
    for word in fillers:
        tx = tx.replace(word, "")
    
    # Remove dates, times, reference numbers
    tx = re.sub(r'\d{2}[/-]\d{2}[/-]\d{2,4}', '', tx)  # Remove dates
    tx = re.sub(r'\d{6,}', '', tx)  # Remove long numbers
    
    # Remove excess whitespace
    tx = " ".join(tx.split())
    
    return tx.strip()

# Get category from keywords
def get_category_from_keywords(transaction):
    """Categorize based on keywords before sending to LLM"""
    if pd.isna(transaction):
        return None
        
    tx_lower = transaction.lower()
    
    # Check each merchant keyword
    for keyword, category in MERCHANT_CATEGORIES.items():
        if keyword in tx_lower:
            return category
    
    # Special cases with multiple conditions
    if ("payment" in tx_lower or "repayment" in tx_lower) and ("loan" in tx_lower or "emi" in tx_lower):
        return "Loans & EMIs"
    
    if "bill" in tx_lower and any(word in tx_lower for word in ["electricity", "water", "gas", "phone"]):
        return "Utilities"
    
    return None

# Create enhanced prompt with examples
def create_categorization_prompt(transactions):
    categories_str = ", ".join(CATEGORIES)
    
    examples = [
        {"transaction": "ICICI BANK LTD EMI REPAYMENT", "category": "Loans & EMIs"},
        {"transaction": "LOAN REPAYMENT REF123456", "category": "Loans & EMIs"},
        {"transaction": "HDFC HOME LOAN PAYMENT", "category": "Loans & EMIs"},
        {"transaction": "RAZORPAY NETFLIX", "category": "Subscriptions"},
        {"transaction": "ZOMATO ORDER 123456", "category": "Food & Dining"},
        {"transaction": "ZOMATO PAYMENT DELHI", "category": "Food & Dining"},
        {"transaction": "UPI/BIGBASKET/PAYMENT", "category": "Groceries"},
        {"transaction": "ATM CASH WITHDRAWAL", "category": "Cash Withdrawal"},
        {"transaction": "AMAZON RETAIL", "category": "Shopping"},
        {"transaction": "UBER TRIP PAYMENT", "category": "Transportation"},
        {"transaction": "ELECTRICITY BILL PAYMENT", "category": "Utilities"},
        {"transaction": "HOSPITAL FEES", "category": "Healthcare"}
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

# Standardize category to ensure consistency
def standardize_category(category):
    """Map similar categories to standard ones"""
    if not category or pd.isna(category):
        return "Miscellaneous"
        
    # First check exact match
    if category in CATEGORIES:
        return category
        
    # Create mapping of keywords to standard categories
    mapping = {
        "food": "Food & Dining",
        "dining": "Food & Dining",
        "restaurant": "Food & Dining",
        "grocery": "Groceries",
        "market": "Groceries",
        "shop": "Shopping",
        "retail": "Shopping",
        "transport": "Transportation",
        "travel": "Travel",
        "holiday": "Travel",
        "vacation": "Travel",
        "utility": "Utilities",
        "bill payment": "Utilities",
        "house": "Housing",
        "rent": "Housing",
        "loan": "Loans & EMIs",
        "emi": "Loans & EMIs",
        "repayment": "Loans & EMIs",
        "credit": "Loans & EMIs",
        "insurance": "Insurance",
        "premium": "Insurance",
        "entertain": "Entertainment",
        "movie": "Entertainment",
        "health": "Healthcare",
        "medical": "Healthcare",
        "education": "Education",
        "tuition": "Education",
        "subscription": "Subscriptions",
        "income": "Income",
        "salary": "Income",
        "investment": "Investments",
        "cash": "Cash Withdrawal",
        "withdrawal": "Cash Withdrawal",
        "transfer": "Transfers",
        "upi": "Transfers",
        "business": "Business Expenses"
    }
    
    # Check for keyword matches
    category_lower = category.lower()
    for key, value in mapping.items():
        if key in category_lower:
            return value
            
    # Fall back to fuzzy matching
    matches = difflib.get_close_matches(category, CATEGORIES, n=1, cutoff=0.6)
    return matches[0] if matches else "Miscellaneous"

# Function to categorize a single transaction
def categorize_transaction(transaction, cache, llm=None):
    """Categorize a single transaction with multiple fallback methods"""
    if pd.isna(transaction) or not transaction:
        return "Miscellaneous", 0.0
    
    # Step 1: Try keyword matching first
    keyword_category = get_category_from_keywords(transaction)
    if keyword_category:
        return keyword_category, 1.0  # High confidence for keyword matches
    
    # Step 2: Normalize the transaction
    normalized_tx = normalize_transaction(transaction)
    if not normalized_tx:
        return "Miscellaneous", 0.0
    
    # Step 3: Check if normalized text is in cache
    if normalized_tx in cache:
        return cache[normalized_tx], 1.0
    
    # Step 4: Use LLM if available
    if llm is not None:
        prompt = create_categorization_prompt([transaction])
        try:
            response = llm.invoke(prompt).strip()
            # Parse response and return category
            for line in response.split('\n'):
                if '→' in line or '->' in line:
                    # Standardize arrows
                    line = line.replace('->', '→')
                    parts = line.split('→', 1)
                    if len(parts) == 2:
                        category = parts[1].strip()
                        std_category = standardize_category(category)
                        cache[normalized_tx] = std_category
                        return std_category, 0.9
        except Exception as e:
            print(f"LLM error: {e}")
    
    # Step 5: Fallback to Miscellaneous
    return "Miscellaneous", 0.0

# Batch processing with better error handling
def process_batch(transactions, llm, cache):
    """Process a batch of transactions with improved handling"""
    results = {}
    confidences = {}
    
    # First check cache and keywords for all transactions
    to_process = []
    for tx in transactions:
        # Skip empty transactions
        if pd.isna(tx) or not tx:
            results[tx] = "Miscellaneous"
            confidences[tx] = 0.0
            continue
            
        # Try keywords first
        keyword_category = get_category_from_keywords(tx)
        if keyword_category:
            results[tx] = keyword_category
            confidences[tx] = 1.0
            continue
            
        # Check cache
        normalized_tx = normalize_transaction(tx)
        if normalized_tx in cache:
            results[tx] = cache[normalized_tx]
            confidences[tx] = 1.0
            continue
            
        # Need to process with LLM
        to_process.append(tx)
    
    # Skip LLM call if all transactions are handled
    if not to_process:
        return results, confidences
    
    # Process with LLM
    try:
        prompt = create_categorization_prompt(to_process)
        response = llm.invoke(prompt).strip()
        
        # Parse response
        parsed_results = {}
        for line in response.split('\n'):
            line = line.strip()
            if '→' in line or '->' in line:
                # Standardize arrows
                line = line.replace('->', '→')
                parts = line.split('→', 1)
                if len(parts) == 2:
                    tx, category = parts[0].strip(' "\''), parts[1].strip()
                    std_category = standardize_category(category)
                    parsed_results[tx] = std_category
        
        # Match results back to original transactions
        for tx in to_process:
            # Try direct match first
            if tx in parsed_results:
                results[tx] = parsed_results[tx]
                confidences[tx] = 0.95
                normalized_tx = normalize_transaction(tx)
                cache[normalized_tx] = parsed_results[tx]
                continue
                
            # Try matching with normalized versions
            normalized_tx = normalize_transaction(tx)
            best_match = None
            best_score = 0
            
            for parsed_tx in parsed_results:
                normalized_parsed = normalize_transaction(parsed_tx)
                score = difflib.SequenceMatcher(None, normalized_tx, normalized_parsed).ratio()
                if score > best_score and score > 0.7:
                    best_score = score
                    best_match = parsed_tx
            
            if best_match:
                results[tx] = parsed_results[best_match]
                confidences[tx] = best_score
                cache[normalized_tx] = parsed_results[best_match]
            else:
                # No good match found
                results[tx] = "Miscellaneous"
                confidences[tx] = 0.0
    
    except Exception as e:
        print(f"Error processing batch: {e}")
        # Fallback to Miscellaneous for errors
        for tx in to_process:
            if tx not in results:
                results[tx] = "Miscellaneous"
                confidences[tx] = 0.0
    
    return results, confidences

# Ensure consistency across similar transactions
def ensure_consistency(categories_dict, confidences_dict):
    """Ensure similar transactions get the same category"""
    
    # Create normalized lookup dictionary
    normalized_dict = {}
    for tx, category in categories_dict.items():
        if pd.isna(tx) or not tx:
            continue
        normalized = normalize_transaction(tx)
        if normalized not in normalized_dict:
            normalized_dict[normalized] = []
        normalized_dict[normalized].append((tx, category, confidences_dict.get(tx, 0.0)))
    
    # Group by merchant name (first word or two)
    merchant_groups = {}
    for tx, category in categories_dict.items():
        if pd.isna(tx) or not tx:
            continue
            
        # Extract potential merchant name (first 1-2 words)
        words = tx.split()
        if not words:
            continue
            
        merchant = words[0].lower()
        if len(words) > 1:
            merchant = f"{merchant} {words[1].lower()}"
        
        if merchant not in merchant_groups:
            merchant_groups[merchant] = []
        merchant_groups[merchant].append((tx, category, confidences_dict.get(tx, 0.0)))
    
    # Apply majority rule within each group
    improved_categories = categories_dict.copy()
    
    for merchant, entries in merchant_groups.items():
        if len(entries) < 2:
            continue
            
        # Count categories for this merchant
        category_counts = {}
        for _, cat, _ in entries:
            if cat not in category_counts:
                category_counts[cat] = 0
            category_counts[cat] += 1
        
        # Skip if there's only one category
        if len(category_counts) < 2:
            continue
            
        # Find dominant category (>60%)
        total = len(entries)
        for cat, count in category_counts.items():
            if count / total > 0.6 and cat != "Miscellaneous":  # 60% threshold and not Miscellaneous
                # Apply to all low-confidence transactions for this merchant
                for tx, current_cat, conf in entries:
                    if conf < 0.8 or current_cat == "Miscellaneous":
                        improved_categories[tx] = cat
                        confidences_dict[tx] = 0.8  # Boost confidence
                break
    
    return improved_categories, confidences_dict

# Process all transactions
def process_transactions(df, batch_size=CONFIG['batch_size']):
    """Process all transactions with consistency checks"""
    # Get unique transactions
    unique_transactions = df[CONFIG['transaction_column']].dropna().unique().tolist()
    print(f"Found {len(unique_transactions)} unique transactions to categorize")
    
    # Load cache
    cache = load_cache()
    print(f"Loaded {len(cache)} cached transactions")
    
    # Initialize LLM
    llm = Ollama(model=CONFIG['model_name'])
    
    # Create batches
    batches = [unique_transactions[i:i+batch_size] for i in range(0, len(unique_transactions), batch_size)]
    print(f"Processing {len(batches)} batches")
    
    # Process all batches
    all_categories = {}
    all_confidences = {}
    
    for i, batch in enumerate(batches):
        print(f"Processing batch {i+1}/{len(batches)} ({len(batch)} transactions)")
        start_time = time.time()
        
        batch_categories, batch_confidences = process_batch(batch, llm, cache)
        all_categories.update(batch_categories)
        all_confidences.update(batch_confidences)
        
        elapsed = time.time() - start_time
        print(f"Batch completed in {elapsed:.2f} seconds")
        
        # Save cache periodically
        if i % 2 == 0 or i == len(batches) - 1:
            save_cache(cache)
    
    # Ensure consistency across categories
    print("Applying consistency checks to improve categorization...")
    all_categories, all_confidences = ensure_consistency(all_categories, all_confidences)
    
    # Save final cache
    save_cache(cache)
    
    # Create results dataframe
    results_df = pd.DataFrame({
        'Transaction': list(all_categories.keys()),
        'Category': list(all_categories.values()),
        'Confidence': [all_confidences.get(tx, 0.0) for tx in all_categories.keys()]
    })
    
    return results_df, all_categories, all_confidences

# Apply categories to the full dataset
def apply_categories(df, category_dict, confidence_dict):
    """Apply categories to the original dataframe"""
    def get_category_info(tx):
        if pd.isna(tx) or not tx:
            return "Miscellaneous", 0.0
            
        # Direct match
        if tx in category_dict:
            return category_dict[tx], confidence_dict.get(tx, 0.0)
            
        # Keyword match
        keyword_cat = get_category_from_keywords(tx)
        if keyword_cat:
            return keyword_cat, 1.0
            
        # Normalized match
        normalized_tx = normalize_transaction(tx)
        for orig_tx, category in category_dict.items():
            if normalize_transaction(orig_tx) == normalized_tx:
                return category, confidence_dict.get(orig_tx, 0.8)
                
        # Fuzzy match
        matches = difflib.get_close_matches(tx, list(category_dict.keys()), n=1, cutoff=0.7)
        if matches:
            match = matches[0]
            confidence = difflib.SequenceMatcher(None, tx, match).ratio()
            return category_dict[match], confidence
            
        return "Miscellaneous", 0.0
    
    # Apply categorization
    categories = []
    confidences = []
    
    for tx in df[CONFIG['transaction_column']]:
        cat, conf = get_category_info(tx)
        categories.append(cat)
        confidences.append(conf)
    
    df['Category'] = categories
    df['Confidence'] = confidences
    
    return df

# Generate category summary
def generate_summary(df):
    """Generate summary statistics for categories"""
    # Basic summary
    summary = df.groupby('Category').agg(
        Count=('Category', 'count'),
        AvgConfidence=('Confidence', 'mean')
    ).sort_values('Count', ascending=False)
    
    # Add amount analysis if available
    amount_cols = [col for col in df.columns if 'amount' in col.lower() or 'value' in col.lower()]
    if amount_cols:
        amount_col = amount_cols[0]
        amount_summary = df.groupby('Category')[amount_col].agg(['sum', 'mean', 'min', 'max'])
        summary = pd.concat([summary, amount_summary], axis=1)
    
    # Low confidence transactions
    low_conf = df[df['Confidence'] < CONFIG['min_confidence']]
    low_conf_summary = low_conf.groupby('Category').size().reset_index(name='LowConfidenceCount')
    
    return {
        'summary': summary,
        'low_confidence': low_conf,
        'low_confidence_summary': low_conf_summary
    }

# Main function
def main():
    # Load data
    try:
        input_file = r'C:\Users\1137862\Desktop\Trnx_Analyser\AA_Data\IM_check_3_page_num_1.csv'
        df = pd.read_csv(input_file, nrows=100)  # Remove nrows for full dataset
        print(f"CSV loaded successfully. {len(df)} rows found.")
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return
    
    # Process transactions
    results_df, category_dict, confidence_dict = process_transactions(df)
    
    # Apply categories to full dataset
    df_categorized = apply_categories(df, category_dict, confidence_dict)
    
    # Generate summary
    summary = generate_summary(df_categorized)
    
    # Save results
    df_categorized.to_csv(CONFIG['output_file'], index=False)
    summary['summary'].to_csv("category_summary.csv")
    summary['low_confidence'].to_csv("low_confidence_transactions.csv")
    
    print(f"\nDone! Results saved to {CONFIG['output_file']}")
    print(f"Category summary saved to category_summary.csv")
    print(f"Low confidence transactions saved to low_confidence_transactions.csv")
    
    # Print statistics
    print("\nCategory Statistics:")
    print(summary['summary'].head(10))
    print(f"\nTotal categories: {len(summary['summary'])}")
    print(f"Average confidence: {df_categorized['Confidence'].mean():.2f}")
    print(f"Low confidence transactions: {len(summary['low_confidence'])} ({len(summary['low_confidence'])/len(df_categorized)*100:.1f}%)")

if __name__ == "__main__":
    main()
