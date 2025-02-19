import time
import pandas as pd

def categorize_transactions(transaction_names, llm, batch_num):
    try:
        start_time = time.time()
        
        # 📝 Create the LLM prompt
        prompt = (
            "Can you categorize the following transactions? "
            "Return output in 'Transaction - Category' format:\n\n"
            + "\n".join(transaction_names)
        )

        # 🧮 Invoke the LLM
        response = llm.invoke(prompt).strip()
        elapsed_time = time.time() - start_time

        print(f"\n⏳ Batch {batch_num}: Processed {len(transaction_names)} transactions in {elapsed_time:.2f} seconds")
        print(f"🔹 RAW RESPONSE from LLM:\n{response}\n")

        # 🧐 Check if response is empty
        if not response:
            print(f"❌ WARNING: Empty response from LLM for batch {batch_num}")
            return pd.DataFrame({'Transaction': transaction_names, 'Category': ['Unknown'] * len(transaction_names), 'Raw_Category_Response': [''] * len(transaction_names)})

        # 📊 Split response by new lines
        response_lines = response.split("\n")
        print(f"🔍 Debug: First 5 response lines: {response_lines[:5]}")

        # ✅ Process responses properly
        transactions = []
        categories = []
        raw_responses = []

        for line in response_lines:
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

        # 🛠️ Print a sample to verify
        print(f"✅ Batch {batch_num} completed in {elapsed_time:.2f} seconds")
        print(f"📊 Sample Output:\n{df.head()}\n")

        return df

    except Exception as e:
        print(f"❌ Error processing batch {batch_num}: {e}")
        return pd.DataFrame({'Transaction': transaction_names, 'Category': ['Unknown'] * len(transaction_names), 'Raw_Category_Response': [''] * len(transaction_names)})

