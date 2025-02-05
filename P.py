import requests
import pandas as pd

# Load PIN codes from a CSV file
input_file = "input_pincodes.csv"  # Replace with your actual file name
output_file = "output_pincode_details.csv"

# Read CSV file
df = pd.read_csv(input_file)

# Ensure column name is correct
if "Pincode" not in df.columns:
    raise ValueError("The input CSV must have a column named 'Pincode'.")

pin_codes = df["Pincode"].astype(str).tolist()

results = []
for pin in pin_codes:
    url = f"https://api.postalpincode.in/pincode/{pin}"
    response = requests.get(url).json()
    if response[0]['Status'] == 'Success':
        district = response[0]['PostOffice'][0]['District']
        state = response[0]['PostOffice'][0]['State']
        results.append([pin, district, state])
    else:
        results.append([pin, "Not Found", "Not Found"])

# Convert to DataFrame
df_output = pd.DataFrame(results, columns=["Pincode", "District", "State"])

# Save results to a new CSV file
df_output.to_csv(output_file, index=False)
print(f"Data saved successfully in {output_file}!")
