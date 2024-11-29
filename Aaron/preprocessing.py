#%%
import numpy as np
import scipy as sp
import pandas as pd
import os

raw_data = pd.read_csv(os.getcwd() + "/stage 1/train_data.csv")

tmp_df = pd.DataFrame()

#%%

# Convert "date" to datetime format
tmp_df['date'] = pd.to_datetime(raw_data['date'])

# Extract year, month, day, and day of the week
# No year column because it is the same as season
raw_data.insert(raw_data.columns.get_loc("date"), "year", tmp_df['date'].dt.year )
raw_data.insert(raw_data.columns.get_loc("date"), "month", tmp_df['date'].dt.month )
raw_data.insert(raw_data.columns.get_loc("date"), "day", tmp_df['date'].dt.day )
# raw_data.insert(raw_data.columns.get_loc("date"), "dow", tmp_df['date'].dt.weekday )

#Encode True/False
raw_data['is_night_game'] = raw_data['is_night_game'].replace({True: 1, False: 0})
raw_data['home_team_win'] = raw_data['home_team_win'].replace({True: 1, False: 0})


#%%

# Data interpolation:
# For pitchers, replace NULL with the median of pitchers from the same team within the same season
# For other numerical data, fill in with the mean from the same team in the same season

# Functions for filling NaNs
def fill_numeric_with_mean(group):
    return group.fillna(group.mean())

# Functions for filling NaNs
def fill_with_mode_or_random(group):
    mode = group.mode()  # Calculate mode
    if not mode.empty:
        return group.fillna(mode.iloc[0])  # Fill with mode if available
    
    # If mode is null, fill with a random existing value in the group
    valid_values = group.dropna().tolist()  # Get non-NaN values
    if valid_values:
        return group.fillna(random.choice(valid_values))  # Fill with a random value
    return group  # If no valid values, leave as-is


# Process home team columns
home_columns = [col for col in raw_data.columns if col.startswith('home_')]
for col in home_columns:
    if raw_data[col].dtype == 'O' or raw_data[col].dtype == 'bool' :  # Categorical column
        raw_data[col] = raw_data.groupby(['home_team_abbr', 'year'])[col].transform(fill_with_mode_or_random)
    else:  # Numeric column
        raw_data[col] = raw_data.groupby(['home_team_abbr', 'year'])[col].transform(fill_numeric_with_mean)

# Process away team columns
away_columns = [col for col in raw_data.columns if col.startswith('away_')]
for col in away_columns:
    if raw_data[col].dtype == 'O' or raw_data[col].dtype == 'bool':  # Categorical column
        raw_data[col] = raw_data.groupby(['away_team_abbr', 'year'])[col].transform(fill_with_mode_or_random)
    else:  # Numeric column
        raw_data[col] = raw_data.groupby(['away_team_abbr', 'year'])[col].transform(fill_numeric_with_mean)

raw_data['is_night_game'] = raw_data.groupby(['year'])['is_night_game'].transform(fill_with_mode_or_random)

#%%
# Target encoding on categorical columns
columns_to_encode = ['home_team_abbr', 'away_team_abbr', 'home_pitcher', 'away_pitcher']

for col in columns_to_encode:
    mean = raw_data.groupby(col)['home_team_win'].mean() 
    tmp_df[col + '_encoded'] = raw_data[col].map(mean)
    raw_data.insert(raw_data.columns.get_loc(col), col + '_encoded', tmp_df[col + '_encoded'], )

#%%
# Drop columns
columns_to_drop = columns_to_encode + ['date', 'season', 'home_team_season', 'away_team_season', 'id']
for col in columns_to_drop:
    raw_data = raw_data.drop(col, axis=1)


#%%
# print(raw_data)

# Get unique count
for col in raw_data.columns:
    print(f"{col} unique count: {raw_data[col].nunique()}")

# Count NaNs in each column
col_nan_count = raw_data.isnull().sum()
print("NaN count per column saved")
col_nan_count.to_csv("NaN_stats.csv")

# Count rows with NaNs
rows_with_nans = raw_data.isnull().any(axis=1).sum()
print(f"rows with NaNs count: {rows_with_nans}")


# # Get column types
# column_types = raw_data.dtypes

# print("Column Types:")
# print(column_types.to_string())


# Calculate correlations of all features with the label
correlations = raw_data.corr()['home_team_win'].drop('home_team_win')  # Drop the self-correlation of 'label'

print("Correlations with label saved")
correlations.to_csv("feature_label_corr.csv")


#%%
# Save cleaned data
raw_data.to_csv("train_data_clean.csv", index=False)
print("Cleaned data saved")

# Count rows with NaNs
rows_with_nans = raw_data.isnull().any(axis=1).sum()
print(f"rows with NaNs count: {rows_with_nans}")

