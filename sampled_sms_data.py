import pandas as pd

# Load the categorized SMS data
input_csv = 'full_sms_categorization_results.csv'
df = pd.read_csv(input_csv)

# Number of samples per category
n_samples = 80

# Sample 80 messages from each category (if available)
sampled_dfs = []
for category in df['category'].unique():
    category_df = df[df['category'] == category]
    sampled_df = category_df.sample(n=min(n_samples, len(category_df)), random_state=42)
    sampled_dfs.append(sampled_df)

# Concatenate all sampled data
sampled_data = pd.concat(sampled_dfs, ignore_index=True)

# Only keep the original message column (assume it's named 'message')
if 'processed_message' in sampled_data.columns:
    sampled_data = sampled_data[['processed_message']]
else:
    raise ValueError("Column 'processed_message' not found in the input CSV.")

# Shuffle the rows randomly
sampled_data = sampled_data.sample(frac=1, random_state=42).reset_index(drop=True)

# Save to a new CSV file
output_csv = 'sampled_sms_data.csv'
sampled_data.to_csv(output_csv, index=False)

print(f"Sampled data saved to {output_csv} with {len(sampled_data)} rows and only the message column.") 