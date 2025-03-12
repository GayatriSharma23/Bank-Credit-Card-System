import time
import difflib
import pandas as pd
from config import CONFIG
from categorizers import (
    normalize_transaction, get_category_from_keywords,
    create_categorization_prompt, standardize_category
)
from data_handlers import load_cache, save_cache

def normalize_transaction_type(type_value):
    """Normalize transaction type to CREDIT or DEBIT"""
    if pd.isna(type_value) or not type_value:
        return "DEBIT"  # Default to DEBIT
        
    type_value = str(type_value).strip().upper()
    
    if type_value in ['CREDIT', 'CR', 'C', 'DEPOSIT', 'RECEIVED']:
        return "CREDIT"
    else:
        return "DEBIT"  # Default everything else to DEBIT

# Batch processing with better error handling and type consideration
def process_batch(transactions, llm, cache):
    """Process a batch of transactions with improved handling including transaction type"""
    results = {}
    confidences = {}
   
    # First check cache and keywords for all transactions
    to_process = []
    for tx_data in transactions:
        tx = tx_data["narration"]
        tx_type = tx_data["type"]
        tx_id = tx_data["id"]
        
        # Skip empty transactions
        if pd.isna(tx) or not tx:
            results[tx_id] = "Miscellaneous"
            confidences[tx_id] = 0.0
            continue
           
        # Try keywords first (now with type consideration)
        keyword_category = get_category_from_keywords(tx, tx_type)
        if keyword_category:
            results[tx_id] = keyword_category
            confidences[tx_id] = 1.0
            continue
           
        # Check cache with composite key
        normalized_tx = normalize_transaction(tx)
        composite_key = f"{normalized_tx}|{tx_type}"
        if composite_key in cache:
            results[tx_id] = cache[composite_key]
            confidences[tx_id] = 1.0
            continue
           
        # Need to process with LLM
        to_process.append(tx_data)
   
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
        for i, tx_data in enumerate(to_process):
            tx = tx_data["narration"]
            tx_type = tx_data["type"]
            tx_id = tx_data["id"]
            
            # Try direct match first
            if tx in parsed_results:
                results[tx_id] = parsed_results[tx]
                confidences[tx_id] = 0.95
                normalized_tx = normalize_transaction(tx)
                composite_key = f"{normalized_tx}|{tx_type}"
                cache[composite_key] = parsed_results[tx]
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
                # Apply type-based validation/correction
                category = parsed_results[best_match]
                
                # Adjust category based on transaction type if needed
                if tx_type == "CREDIT" and not any(income_term in category.lower() for income_term in ["income", "refund", "reimbursement", "gift"]):
                    # If it's a credit but not categorized as income-related, consider adjusting
                    if hasattr(CONFIG, 'TRANSACTION_PATTERNS'):
                        for keyword, pattern_category in CONFIG.TRANSACTION_PATTERNS.items():
                            if keyword in tx.upper() and "Income" in pattern_category:
                                category = pattern_category
                                break
                
                results[tx_id] = category
                confidences[tx_id] = best_score
                composite_key = f"{normalized_tx}|{tx_type}"
                cache[composite_key] = category
            else:
                # No good match found - use default type-based fallback
                if tx_type == "CREDIT":
                    results[tx_id] = "Other Income"
                else:
                    results[tx_id] = "Miscellaneous"
                confidences[tx_id] = 0.0
   
    except Exception as e:
        print(f"Error processing batch: {e}")
        # Fall back to type-based defaults for all remaining transactions
        for tx_data in to_process:
            tx_id = tx_data["id"]
            tx_type = tx_data["type"]
            if tx_id not in results:
                if tx_type == "CREDIT":
                    results[tx_id] = "Other Income"
                else:
                    results[tx_id] = "Miscellaneous"
                confidences[tx_id] = 0.0
               
    return results, confidences
 
def process_transactions(df):
    """Process transactions with type column for categorization"""
    # Initialize LLM
    from langchain_community.llms import Ollama
    llm = Ollama(model=CONFIG['model_name'])
   
    # Load cache
    cache = load_cache()
   
    # Check if required columns exist
    if CONFIG['transaction_column'] not in df.columns:
        print(f"Error: Transaction column '{CONFIG['transaction_column']}' not found in data")
        return df, {}, {}
        
    if CONFIG['type_column'] not in df.columns:
        print(f"Warning: Type column '{CONFIG['type_column']}' not found, using DEBIT as default")
        df[CONFIG['type_column']] = "DEBIT"  # Default to DEBIT if type column missing
    
    # Normalize column values to CREDIT/DEBIT
    df[CONFIG['type_column']] = df[CONFIG['type_column']].apply(normalize_transaction_type)
    
    # Create dictionaries to store results
    category_dict = {}
    confidence_dict = {}
    
    # Get transactions that need categorization
    transactions_to_process = []
    
    for idx, row in df.iterrows():
        tx = row[CONFIG['transaction_column']]
        tx_type = row[CONFIG['type_column']]
        
        if pd.isna(tx) or not tx:
            category_dict[idx] = "Miscellaneous"
            confidence_dict[idx] = 0.0
            continue
            
        normalized_tx = normalize_transaction(tx)
        composite_key = f"{normalized_tx}|{tx_type}"
        
        # Check if in cache already
        if composite_key in cache:
            category_dict[idx] = cache[composite_key]
            confidence_dict[idx] = 1.0  # Full confidence for cache hits
        else:
            # Add to processing batch with ID for tracking
            transactions_to_process.append({
                "id": idx,
                "narration": tx,
                "type": tx_type
            })
    
    # Process in batches
    if transactions_to_process:
        batch_size = CONFIG['batch_size']
        num_batches = len(transactions_to_process) // batch_size + (1 if len(transactions_to_process) % batch_size > 0 else 0)
        
        print(f"Processing {len(transactions_to_process)} transactions in {num_batches} batches...")
        
        start_time = time.time()
        for batch_num in range(num_batches):
            start_idx = batch_num * batch_size
            end_idx = min(start_idx + batch_size, len(transactions_to_process))
            batch = transactions_to_process[start_idx:end_idx]
            
            print(f"Processing batch {batch_num+1}/{num_batches} ({len(batch)} transactions)...")
            batch_results, batch_confidences = process_batch(batch, llm, cache)
            
            # Update results
            category_dict.update(batch_results)
            confidence_dict.update(batch_confidences)
            
            # Save cache periodically
            if batch_num % 5 == 0:
                save_cache(cache)
                
            # Sleep to avoid rate limiting
            time.sleep(0.5)
    
    # Save final cache
    save_cache(cache)
   
    # Create a copy of the input DataFrame with categories added
    result_df = df.copy()
    
    # Apply categories to the DataFrame
    result_df['Category'] = pd.Series(category_dict)
    result_df['Confidence'] = pd.Series(confidence_dict)
   
    # Handle missing categories (should be rare)
    result_df['Category'] = result_df['Category'].fillna("Miscellaneous")
    result_df['Confidence'] = result_df['Confidence'].fillna(0.0)
   
    # Calculate processing statistics
    if transactions_to_process:
        end_time = time.time()
        processing_time = end_time - start_time
        print(f"Processing completed in {processing_time:.2f} seconds")
        
        # Calculate completion rate
        high_conf = (result_df['Confidence'] >= CONFIG['min_confidence']).mean() * 100
        print(f"High confidence categorizations: {high_conf:.1f}%")
   
    # Apply consistency checks (now with type awareness)
    result_df = ensure_consistency(result_df)
    
    return result_df, category_dict, confidence_dict
 
def apply_categories(df, category_dict, confidence_dict):
    """Apply categorization results to the complete DataFrame
   
    Args:
        df (pandas.DataFrame): Original DataFrame
        category_dict (dict): Mapping from index to category
        confidence_dict (dict): Mapping from index to confidence
       
    Returns:
        pandas.DataFrame: DataFrame with categories added
    """
    # Create a copy to avoid modifying the original
    categorized_df = df.copy()
   
    # Apply category and confidence to each row using index
    categorized_df['Category'] = pd.Series(category_dict)
    categorized_df['Confidence'] = pd.Series(confidence_dict)
   
    # Handle missing values
    categorized_df['Category'] = categorized_df['Category'].fillna("Miscellaneous")
    categorized_df['Confidence'] = categorized_df['Confidence'].fillna(0.0)
   
    # Apply consistency checks
    categorized_df = ensure_consistency(categorized_df)
   
    return categorized_df
 
def ensure_consistency(df):
    """Ensure consistency in categorization across similar transactions
   
    Args:
        df (pandas.DataFrame): DataFrame with initial categorizations
       
    Returns:
        pandas.DataFrame: DataFrame with consistent categorizations
    """
    # Group by normalized transaction text AND transaction type
    normalized_map = {}
    for idx, row in df.iterrows():
        tx = row[CONFIG['transaction_column']]
        tx_type = row[CONFIG['type_column']]
        category = row['Category']
        confidence = row['Confidence']
       
        if pd.isna(tx) or not tx:
            continue
           
        normalized_tx = normalize_transaction(tx)
        # Create composite key with transaction type
        composite_key = f"{normalized_tx}|{tx_type}"
       
        if composite_key not in normalized_map:
            normalized_map[composite_key] = []
           
        normalized_map[composite_key].append((idx, category, confidence))
   
    # Find best category for each normalized transaction + type combination
    best_categories = {}
    for composite_key, categories_data in normalized_map.items():
        if len(categories_data) == 1:
            best_categories[categories_data[0][0]] = categories_data[0][1]
            continue
           
        # Count categories and weighted by confidence
        category_scores = {}
        for idx, category, confidence in categories_data:
            if category not in category_scores:
                category_scores[category] = 0
            category_scores[category] += confidence
           
        # Get category with highest score
        best_category = max(category_scores.items(), key=lambda x: x[1])[0]
        
        # Apply to all indices in this group with low confidence
        for idx, category, confidence in categories_data:
            if confidence < CONFIG['min_confidence']:
                best_categories[idx] = best_category
   
    # Apply consistency
    consistent_df = df.copy()
    for idx, category in best_categories.items():
        # Only override if confidence is low
        if consistent_df.loc[idx, 'Confidence'] < CONFIG['min_confidence']:
            consistent_df.loc[idx, 'Category'] = category
            # Increase confidence slightly but mark it as derived
            consistent_df.loc[idx, 'Confidence'] = min(
                CONFIG['min_confidence'],
                consistent_df.loc[idx, 'Confidence'] + 0.1
            )
               
    return consistent_df
