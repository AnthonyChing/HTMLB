#%%
import numpy as np
import scipy as sp
import pandas as pd
from sklearn.preprocessing import TargetEncoder
import os

# data
# raw_data = pd.read_csv(os.getcwd() + "/../stage 1/train_data.csv")
raw_data = pd.read_csv(os.getcwd() + "/partially_filled.csv")
test_data_stage_1 = pd.read_csv(os.getcwd() + "/../stage 1/same_season_test_data.csv")
test_data_stage_2 = pd.read_csv(os.getcwd() + "/../stage 2/2024_test_data.csv")

data = [raw_data, test_data_stage_1, test_data_stage_2]

tmp_df = pd.DataFrame()

#%%
# Drop helper columns
drop = ['prev_pitcher_date', 'pitcher_diff_days']
for col in drop:
    data[0] = data[0].drop(col, axis=1)

# Count NaNs in each column
for i in range(0, 3):
    col_nan_count = data[i].isnull().sum()
    print("NaN count per column saved")
    col_nan_count.to_csv(f"NaN_stats_{i}.csv")
# Count rows with NaNs
for i in range(0, 3):
    rows_with_nans = data[i].isnull().any(axis=1).sum()
    print(f"rows with NaNs count in data_{i}: {rows_with_nans}")

#%%

# Convert "date" to datetime format
data[0]['date'] = pd.to_datetime(data[0]['date'])

# Replace season column with year, since some data in seasons are missing, but none in year are missing
data[0].insert(data[0].columns.get_loc("date"), "year", data[0]['date'].dt.year )

#Encode True/False
for i in range(0, 3):
    data[i]['is_night_game'] = data[i]['is_night_game'].replace({True: 1, False: 0})

data[0]['home_team_win'] = data[0]['home_team_win'].replace({True: 1, False: 0})


#%%

# Data interpolation:
# For pitchers, replace NULL with the median of pitchers from the same team within the same season
# For other numerical data, fill in with the mean from the same team in the same season

# Fill season column in test set, use home_team_season or away_team_season to fill 
# Extract the year (YYYY) from the columns
for i in range(1, 3):
    data[i]['home_year'] = data[i]['home_team_season'].str.extract(r'_(\d{4})')
    data[i]['away_year'] = data[i]['away_team_season'].str.extract(r'_(\d{4})')

    # Fill the 'season' column with the extracted years
    data[i]['season'] = data[i]['season'].fillna(data[i]['home_year']).fillna(data[i]['away_year'])

    # Fill remaining NaNs in 'season' with the mode year
    mode = data[i]['season'].mode()
    data[i]['season'] = data[i]['season'].fillna(mode[0])

    # Drop helper columns if no longer needed
    data[i] = data[i].drop(columns=['home_year', 'away_year'])
    # duplicate season as year
    data[i]['year'] = data[i]['season']

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
for i in range(0, 3):
    home_columns = [col for col in data[i].columns if col.startswith('home_')]
    away_columns = [col for col in data[i].columns if col.startswith('away_')]
    for cols in [home_columns, away_columns]:
        for col in cols:
            if data[i][col].dtype == 'O' or data[i][col].dtype == 'bool' :  # Categorical column
                data[i][col] = data[i].groupby(['home_team_abbr', 'year'])[col].transform(fill_with_mode_or_random)
            else:  # Numeric column
                data[i][col] = data[i].groupby(['home_team_abbr', 'year'])[col].transform(fill_numeric_with_mean)

for i in range(0, 3):
    data[i]['is_night_game'] = data[i].groupby(['year'])['is_night_game'].transform(fill_with_mode_or_random)

# If there are still NaNs, try again with bigger scope, group by teams only
for i in range(0, 3):
    home_columns = [col for col in data[i].columns if col.startswith('home_')]
    away_columns = [col for col in data[i].columns if col.startswith('away_')]
    for cols in [home_columns, away_columns]:
        for col in cols:
            if data[i][col].dtype == 'O' or data[i][col].dtype == 'bool' :  # Categorical column
                data[i][col] = data[i].groupby('home_team_abbr')[col].transform(fill_with_mode_or_random)
            else:  # Numeric column
                data[i][col] = data[i].groupby('home_team_abbr')[col].transform(fill_numeric_with_mean)

#%%
# Drop columns
columns_to_drop = ['date', 'year', 'home_team_season', 'away_team_season', 'id']
for col in columns_to_drop:
    data[0] = data[0].drop(col, axis=1)
columns_to_drop.remove('date')
for col in columns_to_drop:
    data[1] = data[1].drop(col, axis=1)
    data[2] = data[2].drop(col, axis=1)


#%%

# Save unencoded csvs
data[0].to_csv("training_data_filled.csv", index_label='id')
data[1].to_csv("stage_1_filled.csv", index_label='id')
data[2].to_csv("stage_2_filled.csv", index_label='id')


#%%
# Partition training data into training & validation set
# For stage 1
# Partition the DataFrame, data[3] & data[4] are the training & validation set for stage 1
# This produces a validation set that is 10% the size of all the data
before_cutoff = data[0][(data[0]['month'] < 7) | ((data[0]['month'] == 7) & (data[0]['day'] < 20))]
after_cutoff = data[0][(data[0]['month'] > 7) | ((data[0]['month'] == 7) & (data[0]['day'] >= 20))]
data.append(before_cutoff)
data.append(after_cutoff)





#%%
# Target encoding on categorical columns
columns_to_encode = ['home_team_abbr', 'away_team_abbr', 'home_pitcher', 'away_pitcher']

encoder = TargetEncoder(target_type='binary')
target = raw_data['home_team_win']

for col in columns_to_encode:
    x = pd.DataFrame(raw_data[col])
    col_encoded = encoder.fit_transform(x, target)
    # print(type(col_encoded))
    raw_data.insert(raw_data.columns.get_loc(col), col + '_encoded', col_encoded )
    # Save encoding mapping to encode test set
    mapping = raw_data.drop_duplicates(subset=col, keep='first')[[col, col + '_encoded']]
    mapping.head()
    with open(f"{col}_encoding.txt", 'w') as f:
        f.write(mapping.to_string())





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


