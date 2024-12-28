#%%
import numpy as np
import scipy as sp
import pandas as pd
import os


data1 = pd.read_csv(os.getcwd() + "/train_data_2.csv")
data2 = pd.read_csv(os.getcwd() + "/val_data_2.csv")
data = [data1, data2]

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
for i in range(0, 2):
    for col in cols:
        data[i][col] = data[i].groupby(['home_team_abbr', 'season'])[col].transform(fill_with_mode_or_random)

for i in range(0, 2):
    data[i]['is_night_game'] = data[i].groupby(['season'])['is_night_game'].transform(fill_with_mode_or_random)

# If there are still NaNs, try again with bigger scope, group by teams only
for i in range(0, 2):
    for col in cols:
        if data[i][col].dtype == 'O' or data[i][col].dtype == 'bool' :  # Categorical column
            data[i][col] = data[i].groupby('home_team_abbr')[col].transform(fill_with_mode_or_random)


data[0].to_csv("train_data_2_f.csv", index=False)
data[1].to_csv("val_data_2_f.csv", index=False)