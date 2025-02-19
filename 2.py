import pandas as pd
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# Load CSV file
try:
    df = pd.read_csv(r'C:\Users\1137862\Desktop\Trnx_Analyser\AA_Data\IM_check_3_page_num_1.csv')
    print(df.head())
except Exception as e:
    print(f"Error: {e}")

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
def categorize_transactions(transaction_names, llm):
    try:
        # Prompt LLM with batched transactions
        prompt = (
            "Can you categorize the following transactions? Return output in 'Transaction - Category' format:\n\n" 
            + "\n".join(transaction_names)
        )
        response = llm.invoke(prompt).strip()
        response_lines = response.split("\n")
        
        # Convert response to DataFrame
        categories_df = pd.DataFrame({'Transaction vs Category': response_lines})
        categories_df[['Transaction', 'Category']] = categories_df['Transaction vs Category'].str.split(' - ', expand=True)
        return categories_df[['Transaction', 'Category']]
    except Exception as e:
        print(f"Error processing batch: {e}")
        return pd.DataFrame(columns=['Transaction', 'Category'])

# DataFrame to store categorized transactions
categories_df_all = pd.DataFrame(columns=['Transaction', 'Category'])

# Loop through the index list and batch process transactions
for i in range(len(index_list) - 1):
    batch_transactions = unique_transactions[index_list[i]:index_list[i+1]]
    categories_df = categorize_transactions(batch_transactions, llm)
    categories_df_all = pd.concat([categories_df_all, categories_df], ignore_index=True)

# Merge categorized transactions back to original DataFrame
df = df.merge(categories_df_all, left_on=TRANSACTION_COLUMN, right_on='Transaction', how='left').drop(columns=['Transaction'])

# Save categorized transactions
output_file = "categorized_transaction_mixtral.csv"
df.to_csv(output_file, index=False)
print("It's Done, check the file")
