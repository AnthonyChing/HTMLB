import pandas as pd
import os

# Read the CSV file as strings to strip whitespace from all columns
df = pd.read_csv('train_data.csv', dtype=str)

# Strip whitespace from column names
df.columns = [col.strip() for col in df.columns]

# Strip whitespace from each cell in string columns
for col in df.columns:
    df[col] = df[col].str.strip()

# Convert 'date' column to datetime
df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d', errors='coerce')

# If 'season' does not exist, create it from the year of 'date'
if 'season' not in df.columns:
    df['season'] = df['date'].dt.year.astype(float)
else:
    # If 'season' exists but has missing values, fill them from the date
    missing_season_mask = df['season'].isna() | (df['season'] == '')
    # Convert season column to float where possible
    df['season'] = pd.to_numeric(df['season'], errors='coerce')
    df.loc[missing_season_mask, 'season'] = df.loc[missing_season_mask, 'date'].dt.year.astype(float)

# Sort by date
df = df.sort_values(by='date')

# Save the cleaned and sorted DataFrame
df.to_csv('sorted.csv', index=False)

# # Create a directory called 'teams' if it doesn't exist
if not os.path.exists('teams'):
    os.makedirs('teams')

# Get all unique home_team_abbr values
unique_teams = df['home_team_abbr'].unique()

# For each unique team, filter rows and save to separate CSV
for team in unique_teams:
    team_df = df[df['home_team_abbr'] == team]
    team_df.to_csv(f'teams/{team}.csv', index=False)