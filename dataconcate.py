import pandas as pd
import glob
import os

# Path to your CSV files
path = # Replace with your actual folder path
all_files = glob.glob(os.path.join(path, "*.csv"))

# Combine all CSV files into a single DataFrame
df_list = [pd.read_csv(file) for file in all_files]
combined_data = pd.concat(df_list, ignore_index=True)

# Define the output path
output_file = # Replace with your desired output file path

# Save the combined data to a new CSV file
combined_data.to_csv(output_file, index=False)

print(f"Combined CSV file saved to {output_file}")
