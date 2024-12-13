import pandas as pd
# Convert training data
# Load original and new CSV files
original_df = pd.read_csv('train_data.csv')
updates_df = pd.read_csv('../EDA/Train/data-5.0.csv')

# Ensure both DataFrames have the same number of rows
if len(original_df) != len(updates_df):
    raise ValueError("The number of rows in the two files does not match!")

# Update columns of the original DataFrame with values from the new DataFrame
for column in updates_df.columns:
    if column in original_df.columns:
        original_df[column] = updates_df[column]

tmp_df = pd.DataFrame()
tmp_df['date'] = pd.to_datetime(original_df['date'])
# Concatenate the new columns with the original DataFrame
original_df.insert(original_df.columns.get_loc("date"), "year", tmp_df['date'].dt.year )
original_df.insert(original_df.columns.get_loc("date"), "month", tmp_df['date'].dt.month )
original_df.insert(original_df.columns.get_loc("date"), "day", tmp_df['date'].dt.day )
original_df = original_df.sort_values(by=['year', 'month', 'day'], ascending=[True, True, True])
# Save the updated DataFrame to a new CSV file
original_df.to_csv('updated_original.csv', index=False)

# Convert Testing data
original_df = pd.read_csv('same_season_test_data.csv')
updates_df = pd.read_csv('../EDA/Test/test-data-1.0.csv')

# Ensure both DataFrames have the same number of rows
if len(original_df) != len(updates_df):
    raise ValueError("The number of rows in the two files does not match!")

# Update columns of the original DataFrame with values from the new DataFrame
for column in updates_df.columns:
    if column in original_df.columns:
        original_df[column] = updates_df[column]

original_df = original_df.sort_values(by=['season', 'home_team_abbr'], ascending=[True, True])
# Save the updated DataFrame to a new CSV file
original_df.to_csv('updated_test.csv', index=False)