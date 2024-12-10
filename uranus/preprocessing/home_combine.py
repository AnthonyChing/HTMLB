import os
import pandas as pd

# Directory containing the team CSV files
teams_dir = "teams"
output_file = "home_team_combined.csv"

# Initialize an empty list to store dataframes
dataframes = []

# Loop through all files in the directory
for file_name in os.listdir(teams_dir):
    # Check if the file has a .csv extension
    if file_name.endswith(".csv"):
        file_path = os.path.join(teams_dir, file_name)
        # Read the CSV file and add it to the list
        df = pd.read_csv(file_path)
        # Optionally, add a column with the team abbreviation (from the file name)
        # df['team_abbr'] = os.path.splitext(file_name)[0]  # Use the file name (without extension)
        dataframes.append(df)

# Combine all dataframes into one
combined_df = pd.concat(dataframes, ignore_index=True)

# Save the combined dataframe to a new CSV file
combined_df.to_csv(output_file, index=False)

print(f"Combined CSV file saved as {output_file}")
