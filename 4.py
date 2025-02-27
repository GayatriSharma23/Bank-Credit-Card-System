import pandas as pd
import os
import sys
import difflib
from config import CONFIG
from constants import CATEGORIES
from feedback import add_feedback, load_feedback
from main import main as run_main_process

def collect_feedback():
    """
    Interactive mode to collect feedback on low-confidence transactions
    
    Returns:
        int: Number of feedback items collected
    """
    # Check if low confidence file exists
    low_conf_file = os.path.join("output", "low_confidence_transactions.csv")
    if not os.path.exists(low_conf_file):
        print("No low confidence transactions file found.")
        print("Run main processing first or check the output directory.")
        return 0
        
    # Load low confidence transactions
    try:
        df = pd.read_csv(low_conf_file)
    except Exception as e:
        print(f"Error loading low confidence file: {e}")
        return 0
        
    if len(df) == 0:
        print("No low confidence transactions to review.")
        return 0
    
    print("=" * 60)
    print(" Transaction Feedback Collection ")
    print("=" * 60)
    print(f"Found {len(df)} low confidence transactions for review.")
    print("For each transaction, enter the correct category or press Enter to skip.")
    print("Available categories:")
    
    # Display categories in columns
    categories_per_line = 3
    for i in range(0, len(CATEGORIES), categories_per_line):
        categories_subset = CATEGORIES[i:i+categories_per_line]
        print("  " + "  |  ".join(categories_subset))
    
    print("\nType 'exit' to stop, 'stats' to see progress, 'help' for commands")
    
    feedback_count = 0
    reviewed_count = 0
    
    while reviewed_count < len(df):
        row = df.iloc[reviewed_count]
        tx = row[CONFIG['transaction_column']]
        current_cat = row['Category']
        
        # Display transaction details
        print("\n" + "-" * 60)
        print(f"[{reviewed_count+1}/{len(df)}] Transaction: {tx}")
        print(f"Current category: {current_cat} (Confidence: {row['Confidence']:.2f})")
        
        if 'Amount' in row and not pd.isna(row['Amount']):
            print(f"Amount: {row['Amount']}")
            
        if 'Date' in row and not pd.isna(row['Date']):
            print(f"Date: {row['Date']}")
        
        # Get user input
        category = input("Correct category (or command): ").strip()
        
        # Handle commands
        if category.lower() == 'exit':
            break
        elif category.lower() == 'help':
            print("\nCommands:")
            print("  Enter       - Skip this transaction")
            print("  exit        - Stop the feedback session")
            print("  stats       - Show progress statistics")
            print("  help        - Show this help message")
            print("  list        - Show all categories")
            print("  search XXX  - Search for categories containing XXX")
            continue
        elif category.lower() == 'stats':
            print(f"\nProgress: {reviewed_count}/{len(df)} reviewed")
            print(f"Feedback provided: {feedback_count} transactions")
            continue
        elif category.lower() == 'list':
            print("\nAll categories:")
            for i, cat in enumerate(sorted(CATEGORIES)):
                print(f"  {i+1}. {cat}")
            continue
        elif category.lower().startswith('search '):
            search_term = category.lower()[7:]
            matches = [cat for cat in CATEGORIES if search_term in cat.lower()]
            if matches:
                print(f"\nMatching categories for '{search_term}':")
                for match in matches:
                    print(f"  {match}")
            else:
                print(f"No categories found containing '{search_term}'")
            continue
        elif not category:
            reviewed_count += 1
            continue
            
        # Validate category
        if category not in CATEGORIES:
            # Check if user entered a number
            if category.isdigit() and 1 <= int(category) <= len(CATEGORIES):
                category = sorted(CATEGORIES)[int(category)-1]
            else:
                # Find similar categories
                matches = difflib.get_close_matches(category, CATEGORIES, n=3, cutoff=0.6)
                if matches:
                    print(f"'{category}' is not a valid category. Did you mean:")
                    for i, match in enumerate(matches):
                        print(f"  {i+1}. {match}")
                    choice = input("Enter number or press Enter to try again: ").strip()
                    if choice.isdigit() and 1 <= int(choice) <= len(matches):
                        category = matches[int(choice)-1]
                    else:
                        print("No valid selection. Please try again.")
                        continue
                else:
                    print(f"'{category}' is not a valid category.")
                    print("Use 'list' to see all categories or 'search XXX' to find matching categories.")
                    continue
        
        # Add feedback
        print(f"Adding feedback: '{tx}' → {category}")
        add_feedback(tx, category)
        feedback_count += 1
        reviewed_count += 1
    
    print("\n" + "=" * 60)
    print(f"Feedback session completed:")
    print(f"  Reviewed: {reviewed_count} transactions")
    print(f"  Feedback provided: {feedback_count} transactions")
    
    if feedback_count > 0:
        print("\nTo apply this feedback, run the main process again.")
        apply_now = input("Would you like to run the main process now? (y/n): ").strip().lower()
        if apply_now == 'y':
            print("\nRunning main process to apply feedback...")
            run_main_process()
    
    return feedback_count

def batch_feedback(category_map_file):
    """
    Process batch feedback from a CSV file
    
    Args:
        category_map_file (str): Path to CSV file with transactions and categories
        
    Returns:
        int: Number of feedback items processed
    """
    if not os.path.exists(category_map_file):
        print(f"Error: File not found: {category_map_file}")
        return 0
    
    try:
        map_df = pd.read_csv(category_map_file)
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return 0
    
    # Check required columns
    if CONFIG['transaction_column'] not in map_df.columns or 'Category' not in map_df.columns:
        print(f"Error: Required columns '{CONFIG['transaction_column']}' and 'Category' must be present.")
        print(f"Available columns: {', '.join(map_df.columns)}")
        return 0
    
    feedback_count = 0
    invalid_count = 0
    
    print(f"Processing {len(map_df)} feedback entries...")
    
    for _, row in map_df.iterrows():
        tx = row[CONFIG['transaction_column']]
        category = row['Category']
        
        # Skip empty transactions
        if pd.isna(tx) or not tx:
            continue
            
        # Validate category
        if category not in CATEGORIES:
            print(f"Warning: Invalid category '{category}' for transaction '{tx}'")
            invalid_count += 1
            continue
        
        # Add feedback
        add_feedback(tx, category)
        feedback_count += 1
    
    print(f"Batch feedback complete. Added {feedback_count} feedback entries.")
    if invalid_count > 0:
        print(f"Skipped {invalid_count} entries with invalid categories.")
    
    return feedback_count

def review_categorized_transactions():
    """
    Review and correct already categorized transactions
    
    Returns:
        int: Number of corrections made
    """
    output_file = os.path.join("output", CONFIG['output_file'])
    if not os.path.exists(output_file):
        print(f"Error: Categorized transactions file not found: {output_file}")
        return 0
    
    try:
        df = pd.read_csv(output_file)
    except Exception as e:
        print(f"Error loading file: {e}")
        return 0
    
    print("=" * 60)
    print(" Categorized Transactions Review ")
    print("=" * 60)
    
    # Get search criteria
    print("Search for transactions to review:")
    print("1. By category")
    print("2. By transaction text")
    print("3. By date range")
    print("4. By amount range")
    choice = input("Select search method (1-4) or press Enter for all: ").strip()
    
    filtered_df = df.copy()
    
    if choice == "1":
        print("\nAvailable categories:")
        for i, cat in enumerate(sorted(CATEGORIES)):
            print(f"  {i+1}. {cat}")
        cat_choice = input("Enter category number or name: ").strip()
        
        if cat_choice.isdigit() and 1 <= int(cat_choice) <= len(CATEGORIES):
            category = sorted(CATEGORIES)[int(cat_choice)-1]
        else:
            category = cat_choice
            
        if category not in CATEGORIES:
            matches = difflib.get_close_matches(category, CATEGORIES, n=1, cutoff=0.6)
            if matches:
                category = matches[0]
            else:
                print(f"Invalid category: {category}")
                return 0
                
        filtered_df = filtered_df[filtered_df['Category'] == category]
    
    elif choice == "2":
        search_text = input("Enter text to search for in transactions: ").strip().lower()
        filtered_df = filtered_df[filtered_df[CONFIG['transaction_column']].str.lower().str.contains(search_text, na=False)]
    
    elif choice == "3":
        if 'Date' not in filtered_df.columns:
            print("Error: No 'Date' column found in the data.")
            return 0
            
        start_date = input("Enter start date (YYYY-MM-DD) or press Enter for all: ").strip()
        end_date = input("Enter end date (YYYY-MM-DD) or press Enter for all: ").strip()
        
        if start_date:
            filtered_df = filtered_df[pd.to_datetime(filtered_df['Date']) >= pd.to_datetime(start_date)]
        if end_date:
            filtered_df = filtered_df[pd.to_datetime(filtered_df['Date']) <= pd.to_datetime(end_date)]
    
    elif choice == "4":
        if 'Amount' not in filtered_df.columns:
            print("Error: No 'Amount' column found in the data.")
            return 0
            
        min_amount = input("Enter minimum amount or press Enter for all: ").strip()
        max_amount = input("Enter maximum amount or press Enter for all: ").strip()
        
        if min_amount:
            filtered_df = filtered_df[filtered_df['Amount'] >= float(min_amount)]
        if max_amount:
            filtered_df = filtered_df[filtered_df['Amount'] <= float(max_amount)]
    
    if len(filtered_df) == 0:
        print("No transactions found matching your criteria.")
        return 0
    
    print(f"\nFound {len(filtered_df)} transactions. Showing first 100:")
    
    # Display limited results
    display_df = filtered_df.head(100)
    for i, row in display_df.iterrows():
        tx = row[CONFIG['transaction_column']]
        cat = row['Category']
        amount = row.get('Amount', 'N/A')
        date = row.get('Date', 'N/A')
        
        print(f"{i}. {tx} → {cat} | Amount: {amount} | Date: {date}")
    
    corrections = 0
    
    # Get transaction indices to correct
    indices = input("\nEnter indices to correct (comma-separated) or 'all' for all shown: ").strip()
    
    if indices.lower() == 'all':
        to_correct = display_df.index.tolist()
    elif indices:
        try:
            to_correct = [int(idx) for idx in indices.split(',')]
        except ValueError:
            print("Invalid indices. Please enter numbers separated by commas.")
            return 0
    else:
        print("No indices selected. Exiting review.")
        return 0
    
    # Process corrections
    for idx in to_correct:
        if idx not in display_df.index:
            print(f"Warning: Index {idx} not found in results.")
            continue
            
        row = df.loc[idx]
        tx = row[CONFIG['transaction_column']]
        current_cat = row['Category']
        
        print("\n" + "-" * 60)
        print(f"Transaction: {tx}")
        print(f"Current category: {current_cat}")
        
        # Display categories
        print("\nAvailable categories:")
        categories_per_line = 3
        for i in range(0, len(CATEGORIES), categories_per_line):
            categories_subset = CATEGORIES[i:i+categories_per_line]
            print("  " + "  |  ".join(categories_subset))
        
        # Get correction
        category = input("Enter correct category (or Enter to skip): ").strip()
        
        if not category:
            continue
            
        # Validate category
        if category not in CATEGORIES:
            matches = difflib.get_close_matches(category, CATEGORIES, n=1, cutoff=0.6)
            if matches:
                use_match = input(f"Did you mean '{matches[0]}'? (y/n): ").strip().lower()
                if use_match == 'y':
                    category = matches[0]
                else:
                    continue
            else:
                print(f"Invalid category: {category}")
                continue
        
        # Add feedback
        add_feedback(tx, category)
        corrections += 1
    
    print("\n" + "=" * 60)
    print(f"Review completed with {corrections} corrections.")
    
    if corrections > 0:
        print("\nTo apply these corrections, run the main process again.")
        apply_now = input("Would you like to run the main process now? (y/n): ").strip().lower()
        if apply_now == 'y':
            print("\nRunning main process to apply corrections...")
            run_main_process()
    
    return corrections

def main():
    """
    Main function for interactive mode
    """
    print("=" * 60)
    print(" Transaction Categorization - Interactive Mode ")
    print("=" * 60)
    print("1. Collect feedback on low-confidence transactions")
    print("2. Review and correct already categorized transactions")
    print("3. Import batch feedback from CSV")
    print("4. Run main categorization process")
    print("5. Exit")
    
    choice = input("\nSelect an option (1-5): ").strip()
    
    if choice == "1":
        collect_feedback()
    elif choice == "2":
        review_categorized_transactions()
    elif choice == "3":
        file_path = input("Enter path to CSV feedback file: ").strip()
        if file_path:
            batch_feedback(file_path)
    elif choice == "4":
        input_file = input("Enter path to input CSV file (or Enter for default): ").strip()
        if not input_file:
            input_file = None
        run_main_process(input_file)
    elif choice == "5":
        print("Exiting interactive mode.")
        return
    else:
        print("Invalid choice. Please enter a number between 1 and 5.")
        main()  # Recursively call main to try again
    
    # Ask if user wants to continue
    continue_choice = input("\nWould you like to continue in interactive mode? (y/n): ").strip().lower()
    if continue_choice == 'y':
        main()  # Recursively call main to continue
    else:
        print("Exiting interactive mode.")

if __name__ == "__main__":
    main()
