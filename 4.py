import time
import difflib
import pandas as pd
from config import CONFIG
from categorizers import (
    normalize_transaction, get_category_from_keywords, 
    create_categorization_prompt, standardize_category
)
from data_handlers import load_cache, save_cache

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
        # Fall back to Miscellaneous for all remaining transactions
        for tx in to_process:
            if tx not in results:
                results[tx] = "Miscellaneous"
                confidences[tx] = 0.0
                
    return results, confidences

def process_transactions(df):
    """Process all transactions in a DataFrame and return categorization results
    
    Args:
        df (pandas.DataFrame): DataFrame containing transactions
        
    Returns:
        tuple: (processed_df, category_dict, confidence_dict)
    """
    # Initialize LLM
    from langchain_community.llms import Ollama
    llm = Ollama(model=CONFIG['model_name'])
    
    # Load cache
    cache = load_cache()
    
    # Get unique transactions from the DataFrame
    transactions = df[CONFIG['transaction_column']].dropna().unique().tolist()
    print(f"Processing {len(transactions)} unique transactions...")
    
    # Create dictionaries to store results
    category_dict = {}
    confidence_dict = {}
    
    # Process in batches
    batch_size = CONFIG['batch_size']
    num_batches = len(transactions) // batch_size + (1 if len(transactions) % batch_size > 0 else 0)
    
    start_time = time.time()
    for batch_num in range(num_batches):
        start_idx = batch_num * batch_size
        end_idx = min(start_idx + batch_size, len(transactions))
        batch = transactions[start_idx:end_idx]
        
        print(f"Processing batch {batch_num+1}/{num_batches} ({len(batch)} transactions)...")
        batch_results, batch_confidences = process_batch(batch, llm, cache)
        
        # Update results
        category_dict.update(batch_results)
        confidence_dict.update(batch_confidences)
        
        # Save cache periodically
        if batch_num % 5 == 0:
            save_cache(cache)
    
    # Save final cache
    save_cache(cache)
    
    # Create a copy of the input DataFrame with categories added
    result_df = df.copy()
    result_df['Category'] = result_df[CONFIG['transaction_column']].map(category_dict)
    result_df['Confidence'] = result_df[CONFIG['transaction_column']].map(confidence_dict)
    
    # Handle missing categories (should be rare)
    result_df['Category'] = result_df['Category'].fillna("Miscellaneous")
    result_df['Confidence'] = result_df['Confidence'].fillna(0.0)
    
    # Calculate processing statistics
    end_time = time.time()
    processing_time = end_time - start_time
    print(f"Processing completed in {processing_time:.2f} seconds")
    
    # Calculate completion rate
    high_conf = (result_df['Confidence'] >= CONFIG['min_confidence']).mean() * 100
    print(f"High confidence categorizations: {high_conf:.1f}%")
    
    return result_df, category_dict, confidence_dict

def apply_categories(df, category_dict, confidence_dict):
    """Apply categorization results to the complete DataFrame
    
    Args:
        df (pandas.DataFrame): Original DataFrame
        category_dict (dict): Mapping from transaction to category
        confidence_dict (dict): Mapping from transaction to confidence
        
    Returns:
        pandas.DataFrame: DataFrame with categories added
    """
    # Create a copy to avoid modifying the original
    categorized_df = df.copy()
    
    # Apply category and confidence to each row
    categorized_df['Category'] = categorized_df[CONFIG['transaction_column']].map(category_dict)
    categorized_df['Confidence'] = categorized_df[CONFIG['transaction_column']].map(confidence_dict)
    
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
    # Group by normalized transaction text
    normalized_map = {}
    for tx, category, confidence in zip(
            df[CONFIG['transaction_column']], 
            df['Category'], 
            df['Confidence']):
        
        if pd.isna(tx) or not tx:
            continue
            
        normalized_tx = normalize_transaction(tx)
        
        if normalized_tx not in normalized_map:
            normalized_map[normalized_tx] = []
            
        normalized_map[normalized_tx].append((category, confidence))
    
    # Find best category for each normalized transaction
    best_categories = {}
    for normalized_tx, categories in normalized_map.items():
        if len(categories) == 1:
            best_categories[normalized_tx] = categories[0][0]
            continue
            
        # Count categories and weighted by confidence
        category_scores = {}
        for category, confidence in categories:
            if category not in category_scores:
                category_scores[category] = 0
            category_scores[category] += confidence
            
        # Get category with highest score
        best_category = max(category_scores.items(), key=lambda x: x[1])[0]
        best_categories[normalized_tx] = best_category
    
    # Apply consistency
    consistent_df = df.copy()
    for i, tx in enumerate(consistent_df[CONFIG['transaction_column']]):
        if pd.isna(tx) or not tx:
            continue
            
        normalized_tx = normalize_transaction(tx)
        if normalized_tx in best_categories:
            consistent_category = best_categories[normalized_tx]
            
            # Only override if confidence is low
            if consistent_df.loc[i, 'Confidence'] < CONFIG['min_confidence']:
                consistent_df.loc[i, 'Category'] = consistent_category
                # Increase confidence slightly but mark it as derived
                consistent_df.loc[i, 'Confidence'] = min(
                    CONFIG['min_confidence'], 
                    consistent_df.loc[i, 'Confidence'] + 0.1
                )
                
    return consistent_df
