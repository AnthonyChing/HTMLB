#%%
import numpy as np
import scipy as sp
import pandas as pd
import os


X_train = [pd.DataFrame() for _ in range(4)]
X_val =[pd.DataFrame() for _ in range(4)]
for i in range(4):
    X_train[i] = pd.read_csv(os.getcwd() + f"/train_data_33%_tscv_{i}.csv")
    X_val[i] = pd.read_csv(os.getcwd() + f"/val_data_33%_tscv_{i}.csv")

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
cols = ['home_team_abbr', 'away_team_abbr', 'home_pitcher', 'away_pitcher']
for i in range(4):
    for col in cols:
        X_train[i][col] = X_train[i].groupby(['home_team_abbr', 'season'])[col].transform(fill_with_mode_or_random)

for i in range(4):
    X_train[i]['is_night_game'] = X_train[i].groupby(['season'])['is_night_game'].transform(fill_with_mode_or_random)

# If there are still NaNs, try again with bigger scope, group by teams only
for i in range(4):
    for col in cols:
        if X_train[i][col].dtype == 'O' or X_train[i][col].dtype == 'bool' :  # Categorical column
            X_train[i][col] = X_train[i].groupby('home_team_abbr')[col].transform(fill_with_mode_or_random)



# Validation set
cols = ['home_team_abbr', 'away_team_abbr', 'home_pitcher', 'away_pitcher']
for i in range(4):
    for col in cols:
        X_val[i][col] = X_val[i].groupby(['home_team_abbr', 'season'])[col].transform(fill_with_mode_or_random)

for i in range(4):
    X_val[i]['is_night_game'] = X_val[i].groupby(['season'])['is_night_game'].transform(fill_with_mode_or_random)

# If there are still NaNs, try again with bigger scope, group by teams only
for i in range(4):
    for col in cols:
        if X_val[i][col].dtype == 'O' or X_val[i][col].dtype == 'bool' :  # Categorical column
            X_val[i][col] = X_val[i].groupby('home_team_abbr')[col].transform(fill_with_mode_or_random)


for i in range(4):
    X_train[i].to_csv(f"train_data_tscv_{i}_f.csv", index=False)
    X_val[i].to_csv(f"val_data_tscv_{i}_f.csv", index=False)