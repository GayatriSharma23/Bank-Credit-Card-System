import pandas as pd

# Sample DataFrame
# df = pd.DataFrame(...) # Replace with your actual data if available

# Convert column to numeric, if it’s not already numeric, setting non-convertible values to NaN
df['Distance_btw_Branch_LAPCodes'] = pd.to_numeric(df['Distance_btw_Branch_LAPCodes'], errors='coerce')

# Define a function to categorize each distance value
def categorize_distance(distance):
    if pd.isna(distance):
        return None  # or any other placeholder for NaN values
    elif distance < 2:
        return '<2KM'
    elif 2 <= distance <= 8:
        return '2-8KM'
    elif 8 < distance <= 18:
        return '8-18KM'
    elif 18 < distance <= 35:
        return '18-35KM'
    elif 35 < distance <= 82:
        return '35-82KM'
    elif 82 < distance <= 194:
        return '82-194KM'
    elif 194 < distance <= 610:
        return '194-610KM'
    else:
        return '610+'

# Apply the function to each row in the 'Distance_btw_Branch_LAPCodes' column
df['Distance_btw_Branch_LAPCodes_Bucket'] = df['Distance_btw_Branch_LAPCodes'].apply(categorize_distance)

# Display the result
print(df[['Distance_btw_Branch_LAPCodes', 'Distance_btw_Branch_LAPCodes_Bucket']].head())

