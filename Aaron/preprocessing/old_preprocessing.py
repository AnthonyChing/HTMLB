#%%
import numpy as np
import scipy as sp
import pandas as pd
import os

raw_data = pd.read_csv(os.getcwd() + "/../stage 1/train_data.csv")
test_data = pd.read_csv(os.getcwd() + "/../stage 1/same_season_test_data.csv")
test_data_2 = pd.read_csv(os.getcwd() + "/../stage 2/2024_test_data.csv")

data = [raw_data, test_data, test_data_2]

tmp_df = pd.DataFrame()

#%%

# Convert "date" to datetime format
tmp_df['date'] = pd.to_datetime(data[0]['date'])

# Extract year, month, day, and day of the week
# No year column because it is the same as season
data[0].insert(raw_data.columns.get_loc("date"), "year", tmp_df['date'].dt.year )
# raw_data.insert(raw_data.columns.get_loc("date"), "month", tmp_df['date'].dt.month )
# raw_data.insert(raw_data.columns.get_loc("date"), "day", tmp_df['date'].dt.day )
# raw_data.insert(raw_data.columns.get_loc("date"), "dow", tmp_df['date'].dt.weekday )

#Encode True/False
for i in range(0, 2):
    data[i]['is_night_game'] = data[i]['is_night_game'].replace({True: 1, False: 0})

data[0]['home_team_win'] = data[0]['home_team_win'].replace({True: 1, False: 0})

# Count NaNs in each column
for i in range(0, 2):
    col_nan_count = data[i].isnull().sum()
    print("NaN count per column saved")
    col_nan_count.to_csv(f"NaN_stats_{i}.csv")


# Rows in test_set without season 
# nan_rows = data[1][data[1]['season'].isna()]
# nan_rows.to_csv("test_set_no_season.csv")


#%%

# Data interpolation:
# For pitchers, replace NULL with the median of pitchers from the same team within the same season
# For other numerical data, fill in with the mean from the same team in the same season

# Fill season column in test set, use home_team_season or away_team_season to fill, (there aren't rows with both missing) 
# Extract the year (YYYY) from the columns
data[1]['home_year'] = data[1]['home_team_season'].str.extract(r'_(\d{4})')
data[1]['away_year'] = data[1]['away_team_season'].str.extract(r'_(\d{4})')


# Fill the 'season' column with the extracted years
data[1]['season'] = data[1]['season'].fillna(data[1]['home_year']).fillna(data[1]['away_year'])

# Fill remaining NaNs in 'season' with the mode year
mode = data[1]['season'].mode()
data[1]['season'] = data[1]['season'].fillna(mode[0])

# Drop helper columns if no longer needed
data[1] = data[1].drop(columns=['home_year', 'away_year'])
# duplicate season as year
data[1]['year'] = data[1]['season']


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


# Process home & away team columns
for i in [0,1]:
    home_columns = [col for col in data[i].columns if col.startswith('home_')]
    away_columns = [col for col in data[i].columns if col.startswith('away_')]
    for cols in [home_columns, away_columns]:
        for col in cols:
            if data[i][col].dtype == 'O' or data[i][col].dtype == 'bool' :  # Categorical column
                data[i][col] = data[i].groupby(['home_team_abbr', 'year'])[col].transform(fill_with_mode_or_random)
            else:  # Numeric column
                data[i][col] = data[i].groupby(['home_team_abbr', 'year'])[col].transform(fill_numeric_with_mean)

for i in [0,1]:
    data[i]['is_night_game'] = data[i].groupby(['year'])['is_night_game'].transform(fill_with_mode_or_random)

# If there are still NaNs, try again with bigger scope, group by teams only
for i in [0,1]:
    home_columns = [col for col in data[i].columns if col.startswith('home_')]
    away_columns = [col for col in data[i].columns if col.startswith('away_')]
    for cols in [home_columns, away_columns]:
        for col in cols:
            if data[i][col].dtype == 'O' or data[i][col].dtype == 'bool' :  # Categorical column
                data[i][col] = data[i].groupby('home_team_abbr')[col].transform(fill_with_mode_or_random)
            else:  # Numeric column
                data[i][col] = data[i].groupby('home_team_abbr')[col].transform(fill_numeric_with_mean)

#%%
# Target encoding on categorical columns
columns_to_encode = ['home_team_abbr', 'away_team_abbr', 'home_pitcher', 'away_pitcher']

mappings = {}
for i in [0,1]:
    for col in columns_to_encode:
        if i == 0:
            mean = data[i].groupby(col)['home_team_win'].mean() 
            mappings[col] = mean 
            tmp_df[col + '_encoded'] = data[i][col].map(mean)
            data[i].insert(data[i].columns.get_loc(col), col + '_encoded', tmp_df[col + '_encoded'], )
        else:
            tmp_df[col + '_encoded'] = data[i][col].map(mappings[col])
            data[i].insert(data[i].columns.get_loc(col), col + '_encoded', tmp_df[col + '_encoded'], )


# Fill NaNs again in test set
home_columns = [col for col in data[i].columns if col.startswith('home_')]
away_columns = [col for col in data[i].columns if col.startswith('away_')]
for cols in [home_columns, away_columns]:
    for col in cols:
        if data[i][col].dtype == 'O' or data[i][col].dtype == 'bool' :  # Categorical column
            data[i][col] = data[i].groupby('home_team_abbr')[col].transform(fill_with_mode_or_random)
        else:  # Numeric column
            data[i][col] = data[i].groupby('home_team_abbr')[col].transform(fill_numeric_with_mean)


#%%
# Replace season with year
for i in [0,1]:
    data[i]['season'] = data[i]['year']
    # Drop columns
    columns_to_drop = columns_to_encode + ['year', 'home_team_season', 'away_team_season', 'id']
    for col in columns_to_drop:
        data[i] = data[i].drop(col, axis=1)
data[0] = data[0].drop('date', axis=1)


#%%
# print(raw_data)

# # Get unique count
# for col in raw_data.columns:
#     print(f"{col} unique count: {raw_data[col].nunique()}")

# # Count NaNs in each column
# col_nan_count = raw_data.isnull().sum()
# print("NaN count per column saved")
# col_nan_count.to_csv("NaN_stats.csv")

# # Count rows with NaNs
# rows_with_nans = raw_data.isnull().any(axis=1).sum()
# print(f"rows with NaNs count: {rows_with_nans}")


# # Get column types
# column_types = raw_data.dtypes

# print("Column Types:")
# print(column_types.to_string())


# # Calculate correlations of all features with the label
# correlations = raw_data.corr()['home_team_win'].drop('home_team_win')  # Drop the self-correlation of 'label'

# print("Correlations with label saved")
# correlations.to_csv("feature_label_corr.csv")


#%%
# Save cleaned data
data[0].to_csv("train_data_clean.csv", index=False)
data[1].to_csv("test_data_clean.csv", index=False)
print("Cleaned data saved")

# Count rows with NaNs
rows_with_nans = data[0].isnull().any(axis=1).sum()
print(f"rows with NaNs count in training data: {rows_with_nans}")

rows_with_nans = data[1].isnull().any(axis=1)
print(f"rows with NaNs count in test data: {rows_with_nans.sum()}")
data[1][rows_with_nans].to_csv("rows_with_nans_test.csv")

# Count NaNs in each column
for i in [0,1]:
    col_nan_count = data[i].isnull().sum()
    print("NaN count per column saved")
    col_nan_count.to_csv(f"NaN_stats_{i}.csv")


# %%
