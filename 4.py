import time
import difflib
import pandas as pd
import numpy as np
from config import CONFIG
from constants import CATEGORIES
from categorizers import (
    normalize_transaction, get_category_from_keywords, 
    create_categorization_prompt, standardize_category
)
from data_handlers import load_cache, save_cache
from feedback import load_feedback

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
        # Fallback: assign Miscellaneous to all remaining transactions
        for tx in to_process:
            if tx not in results:
                results[tx] = "Miscellaneous"
                confidences[tx] = 0.0
    
    return results, confidences

def apply_feedback_rules(df):
    """
    Apply user feedback to categorize transactions
    
    Args:
        df (DataFrame): DataFrame containing transactions
        
    Returns:
        DataFrame: Updated DataFrame with feedback-based categories
    """
    # Load feedback data
    feedback_data = load_feedback()
    if not feedback_data:
        print("No feedback data available to apply")
        return df
    
    # Create a copy to avoid modifying the original
    result_df = df.copy()
    
    # Add columns for category and confidence if they don't exist
    if 'Category' not in result_df.columns:
        result_df['Category'] = None
    if 'Confidence' not in result_df.columns:
        result_df['Confidence'] = 0.0
    if 'Rule' not in result_df.columns:
        result_df['Rule'] = None
    
    # Track stats for reporting
    feedback_count = 0
    
    # Apply exact matches first
    for tx, category in feedback_data.items():
        # Find exact matches
        mask = result_df[CONFIG['transaction_column']] == tx
        matches = mask.sum()
        
        if matches > 0:
            # Update categories and set high confidence
            result_df.loc[mask, 'Category'] = category
            result_df.loc[mask, 'Confidence'] = 1.0
            result_df.loc[mask, 'Rule'] = "Feedback: Exact match"
            feedback_count += matches
    
    # Apply fuzzy matches for remaining uncategorized or low confidence transactions
    if CONFIG.get('use_fuzzy_feedback', True):
        uncategorized = result_df[(result_df['Category'].isna()) | 
                                 (result_df['Confidence'] < CONFIG.get('confidence_threshold', 0.7))]
        
        for idx, row in uncategorized.iterrows():
            tx = row[CONFIG['transaction_column']]
            normalized_tx = normalize_transaction(tx)
            
            # Find best match in feedback
            best_match = None
            best_score = 0
            
            for feedback_tx in feedback_data.keys():
                normalized_feedback = normalize_transaction(feedback_tx)
                score = difflib.SequenceMatcher(None, normalized_tx, normalized_feedback).ratio()
                if score > best_score and score > 0.8:  # Higher threshold for fuzzy feedback
                    best_score = score
                    best_match = feedback_tx
            
            if best_match:
                result_df.loc[idx, 'Category'] = feedback_data[best_match]
                result_df.loc[idx, 'Confidence'] = min(0.95, best_score)  # Cap at 0.95
                result_df.loc[idx, 'Rule'] = f"Feedback: Fuzzy match ({best_score:.2f})"
                feedback_count += 1
    
    print(f"Applied feedback to {feedback_count} transactions")
    return result_df

def process_transactions_batch(df, llm, batch_size=50, max_retries=3):
    """
    Process transactions in batches to avoid context limit issues
    
    Args:
        df (DataFrame): DataFrame containing transactions
        llm: Language model instance
        batch_size (int): Number of transactions to process in each batch
        max_retries (int): Maximum number of retries for failed batches
        
    Returns:
        DataFrame: Processed DataFrame with categories and confidence scores
    """
    result_df = df.copy()
    
    # Add columns if they don't exist
    if 'Category' not in result_df.columns:
        result_df['Category'] = None
    if 'Confidence' not in result_df.columns:
        result_df['Confidence'] = 0.0
    if 'Rule' not in result_df.columns:
        result_df['Rule'] = None
    
    # Load the categorization cache
    cache = load_cache()
    
    # Filter for uncategorized transactions
    to_process = result_df[(result_df['Category'].isna()) | 
                          (result_df['Confidence'] < CONFIG.get('confidence_threshold', 0.7))]
    
    if len(to_process) == 0:
        print("No transactions require processing")
        return result_df
    
    print(f"Processing {len(to_process)} transactions in batches of {batch_size}...")
    
    # Process in batches
    total_processed = 0
    tx_column = CONFIG['transaction_column']
    
    # Split into batches
    for i in range(0, len(to_process), batch_size):
        batch = to_process.iloc[i:i+batch_size]
        transactions = batch[tx_column].tolist()
        
        print(f"Processing batch {i//batch_size + 1}/{(len(to_process)-1)//batch_size + 1} ({len(transactions)} transactions)")
        
        # Process with retries
        retries = 0
        success = False
        
        while not success and retries < max_retries:
            try:
                results, confidences = process_batch(transactions, llm, cache)
                success = True
            except Exception as e:
                retries += 1
                print(f"Batch processing failed (attempt {retries}/{max_retries}): {e}")
                time.sleep(2)  # Wait before retry
                
                if retries == max_retries:
                    # Final fallback: assign Miscellaneous
                    print("Max retries reached, using fallback categorization")
                    results = {tx: "Miscellaneous" for tx in transactions}
                    confidences = {tx: 0.0 for tx in transactions}
        
        # Update results
        for tx in transactions:
            if tx in results:
                mask = result_df[tx_column] == tx
                result_df.loc[mask, 'Category'] = results[tx]
                result_df.loc[mask, 'Confidence'] = confidences[tx]
                result_df.loc[mask, 'Rule'] = "LLM" if confidences[tx] > 0.0 else "Fallback"
        
        total_processed += len(transactions)
        
        # Save cache periodically
        if i % (batch_size * 5) == 0 and i > 0:
            save_cache(cache)
            print(f"Progress: {total_processed}/{len(to_process)} transactions processed")
    
    # Final cache save
    save_cache(cache)
    
    print(f"Batch processing complete. Processed {total_processed} transactions.")
    return result_df

def apply_consistency_rules(df):
    """
    Apply consistency rules to ensure similar transactions have consistent categories
    
    Args:
        df (DataFrame): DataFrame containing categorized transactions
        
    Returns:
        DataFrame: DataFrame with consistent categories
    """
    result_df = df.copy()
    consistency_count = 0
    
    # Group similar transactions
    tx_column = CONFIG['transaction_column']
    transactions = result_df[tx_column].tolist()
    
    # For each transaction
    for i, tx1 in enumerate(transactions):
        # Skip already high-confidence transactions
        if result_df.iloc[i]['Confidence'] >= 0.9:
            continue
            
        # Find similar transactions with higher confidence
        similarities = []
        for j, tx2 in enumerate(transactions):
            if i == j or result_df.iloc[j]['Confidence'] < 0.9:
                continue
                
            # Calculate similarity
            norm_tx1 = normalize_transaction(tx1)
            norm_tx2 = normalize_transaction(tx2)
            similarity = difflib.SequenceMatcher(None, norm_tx1, norm_tx2).ratio()
            
            if similarity >= 0.85:
                similarities.append((similarity, j))
        
        # Sort by similarity (highest first)
        similarities.sort(reverse=True)
        
        # Apply most similar high-confidence category if available
        if similarities:
            best_idx = similarities[0][1]
            best_similarity = similarities[0][0]
            best_category = result_df.iloc[best_idx]['Category']
            
            result_df.iloc[i, result_df.columns.get_loc('Category')] = best_category
            result_df.iloc[i, result_df.columns.get_loc('Confidence')] = min(0.85, best_similarity)
            result_df.iloc[i, result_df.columns.get_loc('Rule')] = f"Consistency ({best_similarity:.2f})"
            consistency_count += 1
    
    print(f"Applied consistency rules to {consistency_count} transactions")
    return result_df

def identify_low_confidence(df):
    """
    Identify transactions with low confidence for review
    
    Args:
        df (DataFrame): DataFrame containing categorized transactions
        
    Returns:
        DataFrame: DataFrame containing low confidence transactions
    """
    # Define threshold from config
    threshold = CONFIG.get('confidence_threshold', 0.7)
    
    # Filter low confidence transactions
    low_conf_df = df[df['Confidence'] < threshold].copy()
    
    # Sort by confidence (ascending)
    if len(low_conf_df) > 0:
        low_conf_df = low_conf_df.sort_values('Confidence')
    
    print(f"Identified {len(low_conf_df)} low confidence transactions")
    return low_conf_df

def process_transactions(df, llm):
    """
    Main processing function to categorize transactions
    
    Args:
        df (DataFrame): DataFrame containing transactions
        llm: Language model instance
        
    Returns:
        DataFrame: Processed DataFrame with categories
        DataFrame: Low confidence transactions for review
    """
    print(f"Processing {len(df)} transactions...")
    
    # Apply feedback first
    df = apply_feedback_rules(df)
    
    # Process remaining transactions in batches
    df = process_transactions_batch(df, llm)
    
    # Apply consistency rules
    df = apply_consistency_rules(df)
    
    # Identify low confidence transactions
    low_conf_df = identify_low_confidence(df)
    
    # Ensure all categories are valid
    df['Category'] = df['Category'].apply(
        lambda x: x if x in CATEGORIES else "Miscellaneous"
    )
    
    # Fill any remaining NaNs
    df['Category'] = df['Category'].fillna("Miscellaneous")
    df['Confidence'] = df['Confidence'].fillna(0.0)
    
    # Generate summary statistics
    category_counts = df['Category'].value_counts()
    print("\nCategory Distribution:")
    for category, count in category_counts.items():
        print(f"  {category}: {count} ({count/len(df)*100:.1f}%)")
    
    print(f"\nAverage confidence: {df['Confidence'].mean():.2f}")
    print(f"Transactions requiring review: {len(low_conf_df)} ({len(low_conf_df)/len(df)*100:.1f}%)")
    
    return df, low_conf_df
  
