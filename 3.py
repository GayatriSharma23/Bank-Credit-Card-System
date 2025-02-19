import time
import pandas as pd
from langchain_community.llms import Ollama

# Load CSV file
try:
    df = pd.read_csv(r'C:\Users\1137862\Desktop\Trnx_Analyser\AA_Data\IM_check_3_page_num_1.csv')
    print(f"✅ CSV loaded successfully. {len(df)} rows found.")
    print(df.head())
except Exception as e:
    print(f"❌ Error loading CSV: {e}")
    exit()

TRANSACTION_COLUMN = 'Narration'

# Initialize Ollama LLM
llm = Ollama(model="trnx_analyzer_mixtral:latest")

# Function to generate index list for batch processing
def hop(start, stop, step):
    for i in range(start, stop, step):
        yield i
    yield stop

# Get unique transactions & create batch indices
unique_transactions = df[TRANSACTION_COLUMN].dropna().unique().tolist()
index_list = list(hop(0, len(unique_transactions), 30))  # Adjust batch size as needed

# Function to categorize transactions in batches
def categorize_transactions(transaction_names, llm, batch_num):
    print(f"\n🔹 Starting batch {batch_num} with {len(transaction_names)} transactions...")

    try:
        start_time = time.time()

        # 📝 Create the LLM prompt
        prompt = (
            "Can you categorize the following transactions? "
            "Return output in 'Transaction - Category' format:\n\n"
            + "\n".join(transaction_names)
        )
        
        # 🏁 Print prompt preview
        print(f"📩 Prompt sent to LLM (first 5 transactions): {transaction_names[:5]}")

        # 🧮 Invoke the LLM
        response = llm.invoke(prompt).strip()

        elapsed_time = time.time() - start_time
        print(f"✅ Batch {batch_num} completed in {elapsed_time:.2f} seconds")

        # 📊 Debug raw response
        if not response:
            print(f"❌ WARNING: Empty response from LLM for batch {batch_num}")
            return pd.DataFrame({'Transaction': transaction_names, 'Category': ['Unknown'] * len(transaction_names), 'Raw_Category_Response': [''] * len(transaction_names)})

        print(f"🔍 Raw LLM Response (First 5 lines):\n{response.splitlines()[:5]}")

        # 📊 Process responses properly
        transactions = []
        categories = []
        raw_responses = []

        for line in response.split("\n"):
            if " - " in line:
                txn, cat = line.split(" - ", 1)
                transactions.append(txn.strip())
                categories.append(cat.strip())
            else:
                transactions.append(line.strip())
                categories.append("Unknown")  # Assign "Unknown" for missing categories
            
            raw_responses.append(line.strip())

        # 📝 Create DataFrame
        df = pd.DataFrame({
            'Transaction': transactions,
            'Category': categories,
            'Raw_Category_Response': raw_responses
        })

        # ✅ Print a sample to verify
        print(f"📊 Sample Output (first 3 rows):\n{df.head(3)}\n")
        return df

    except Exception as e:
        print(f"❌ Error processing batch {batch_num}: {e}")
        return pd.DataFrame({'Transaction': transaction_names, 'Category': ['Unknown'] * len(transaction_names), 'Raw_Category_Response': [''] * len(transaction_names)})

# DataFrame to store categorized transactions
categories_df_all = pd.DataFrame(columns=['Transaction', 'Category', 'Raw_Category_Response'])

# Loop through the index list and batch process transactions
for i in range(len(index_list) - 1):
    print(f"\n🚀 Processing batch {i+1}/{len(index_list)-1}...")
    
    batch_transactions = unique_transactions[index_list[i]:index_list[i+1]]
    
    if not batch_transactions:
        print(f"⚠️ Skipping empty batch {i+1}")
        continue

    categories_df = categorize_transactions(batch_transactions, llm, i+1)
    
    categories_df_all = pd.concat([categories_df_all, categories_df], ignore_index=True)

print("\n✅ All batches processed. Merging results into the main DataFrame...")

# Merge categorized transactions back to the original DataFrame
df = df.merge(categories_df_all, left_on=TRANSACTION_COLUMN, right_on='Transaction', how='left').drop(columns=['Transaction'])

# Save the output
output_file = "categorized_transaction_mixtral.csv"
df.to_csv(output_file, index=False)

print(f"\n🎉 Done! Check the output file: {output_file}")
