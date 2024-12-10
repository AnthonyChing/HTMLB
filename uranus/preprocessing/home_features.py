import os
import pandas as pd

# Define the directory containing team CSV files
teams_dir = 'teams'

# List of columns to fill by averaging the previous 1 game and next 1 game
rolling_columns = [
    'home_batting_batting_avg_10RA',
    'home_batting_onbase_perc_10RA',
    'home_batting_onbase_plus_slugging_10RA',
    'home_batting_leverage_index_avg_10RA',
    'home_batting_RBI_10RA',
    'home_pitching_earned_run_avg_10RA',
    'home_pitching_SO_batters_faced_10RA',
    'home_pitching_H_batters_faced_10RA',
    'home_pitching_BB_batters_faced_10RA',
    'home_pitcher_earned_run_avg_10RA',
    'home_pitcher_SO_batters_faced_10RA',
    'home_pitcher_H_batters_faced_10RA',
    'home_pitcher_BB_batters_faced_10RA',
    'home_team_errors_mean',
    'home_team_errors_std',
    'home_team_errors_skew',
    'home_team_spread_mean',
    'home_team_spread_std',
    'home_team_spread_skew',
    'home_team_wins_mean',
    'home_team_wins_std',
    'home_team_wins_skew',
    'home_batting_batting_avg_mean',
    'home_batting_batting_avg_std',
    'home_batting_batting_avg_skew',
    'home_batting_onbase_perc_mean',
    'home_batting_onbase_perc_std',
    'home_batting_onbase_perc_skew',
    'home_batting_onbase_plus_slugging_mean',
    'home_batting_onbase_plus_slugging_std',
    'home_batting_onbase_plus_slugging_skew',
    'home_batting_leverage_index_avg_mean',
    'home_batting_leverage_index_avg_std',
    'home_batting_leverage_index_avg_skew',
    'home_batting_wpa_bat_mean',
    'home_batting_wpa_bat_std',
    'home_batting_wpa_bat_skew',
    'home_batting_RBI_mean',
    'home_batting_RBI_std',
    'home_batting_RBI_skew',
    'home_pitching_earned_run_avg_mean',
    'home_pitching_earned_run_avg_std',
    'home_pitching_earned_run_avg_skew',
    'home_pitching_SO_batters_faced_mean',
    'home_pitching_SO_batters_faced_std',
    'home_pitching_SO_batters_faced_skew',
    'home_pitching_H_batters_faced_mean',
    'home_pitching_H_batters_faced_std',
    'home_pitching_H_batters_faced_skew',
    'home_pitching_BB_batters_faced_mean',
    'home_pitching_BB_batters_faced_std',
    'home_pitching_BB_batters_faced_skew',
    'home_pitching_leverage_index_avg_mean',
    'home_pitching_leverage_index_avg_std',
    'home_pitching_leverage_index_avg_skew',
    'home_pitching_wpa_def_mean',
    'home_pitching_wpa_def_std',
    'home_pitching_wpa_def_skew',
    'home_pitcher_earned_run_avg_mean',
    'home_pitcher_earned_run_avg_std',
    'home_pitcher_earned_run_avg_skew',
    'home_pitcher_SO_batters_faced_mean',
    'home_pitcher_SO_batters_faced_std',
    'home_pitcher_SO_batters_faced_skew',
    'home_pitcher_H_batters_faced_mean',
    'home_pitcher_H_batters_faced_std',
    'home_pitcher_H_batters_faced_skew',
    'home_pitcher_BB_batters_faced_mean',
    'home_pitcher_BB_batters_faced_std',
    'home_pitcher_BB_batters_faced_skew',
    'home_pitcher_leverage_index_avg_mean',
    'home_pitcher_leverage_index_avg_std',
    'home_pitcher_leverage_index_avg_skew',
    'home_pitcher_wpa_def_mean',
    'home_pitcher_wpa_def_std',
    'home_pitcher_wpa_def_skew'
]

def process_team_file(team_path):
    # Read the CSV file as strings to ensure consistent stripping
    df = pd.read_csv(team_path, dtype=str)

    # Strip whitespace from column names
    df.columns = [col.strip() for col in df.columns]

    # Strip whitespace from each cell in string columns
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].str.strip()

    # Convert 'date' column to datetime
    df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d', errors='coerce')

    # Ensure 'season' column exists and is numeric
    if 'season' not in df.columns:
        df['season'] = df['date'].dt.year.astype(float)
    else:
        df['season'] = pd.to_numeric(df['season'], errors='coerce')
        # Fill missing 'season' values with the year from 'date'
        missing_season_mask = df['season'].isna() | (df['season'] == '')
        df.loc[missing_season_mask, 'season'] = df.loc[missing_season_mask, 'date'].dt.year.astype(float)

    # Sort by 'season' and 'date' to ensure chronological order within each team-season
    df = df.sort_values(by=['season', 'date']).reset_index(drop=True)

    # -----------------------
    # Compute home_team_rest
    # -----------------------
    # Initialize 'home_team_rest' with 0.0 for the first game
    df['home_team_rest'] = 0.0

    # Calculate difference in days between current game and previous game
    df['prev_date'] = df['date'].shift(1)
    df['diff_days'] = (df['date'] - df['prev_date']).dt.days

    # For the first game, fill NaN with 0
    df['diff_days'] = df['diff_days'].fillna(0)

    # Apply the rule: if diff_days >= 365, set rest to 1.0, else to the actual difference
    df['home_team_rest'] = df['diff_days'].apply(lambda x: 1.0 if x >= 365 else float(x))

    # Drop temporary columns
    df.drop(columns=['prev_date', 'diff_days'], inplace=True)

    # -------------------------
    # Compute home_pitcher_rest
    # -------------------------
    # Initialize 'home_pitcher_rest' with 1.0 for the first appearance
    df['home_pitcher_rest'] = 1.0

    # Ensure 'home_pitcher' column exists
    if 'home_pitcher' not in df.columns:
        raise KeyError(f"'home_pitcher' column not found in {team_path}")

    # Group by 'home_pitcher' and calculate difference in days between appearances
    df['prev_pitcher_date'] = df.groupby('home_pitcher')['date'].shift(1)
    df['pitcher_diff_days'] = (df['date'] - df['prev_pitcher_date']).dt.days

    # Fill NaN (first appearance) with 60 to apply the rest rule (x >= 60 set to 1.0)
    df['pitcher_diff_days'] = df['pitcher_diff_days'].fillna(60)

    # Apply the rule: if pitcher_diff_days >=60, set rest to 1.0, else to the actual difference
    df['home_pitcher_rest'] = df['pitcher_diff_days'].apply(lambda x: 1.0 if x >= 60 else float(x))

    # Drop temporary columns
    df.drop(columns=['prev_pitcher_date', 'pitcher_diff_days'], inplace=True)

    # -----------------------------------------------
    # Compute Rolling Averages for Specified Columns
    # -----------------------------------------------
    # Ensure rolling columns are numeric
    for col in rolling_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

   # Compute average of the previous and next values per season, filling missing data
    group_keys = ['season']
    for col in rolling_columns:
        df[col] = df.groupby(group_keys)[col].transform(
        lambda x: x.fillna((x.shift(1) + x.shift(-1)) / 2)
    )

    # For the first games of the season, if there's still NaN due to lack of data, 
    # fill using the closest available data from that same season.
    # Using transform with .bfill() ensures index compatibility
    for col in rolling_columns:
        df[col] = df.groupby('season')[col].transform(lambda g: g.bfill())

    # Optional: Fill any remaining NaNs with 0.0 or another default value
    # Uncomment the following line if you wish to fill remaining NaNs
    # df[rolling_columns] = df[rolling_columns].fillna(0.0)

    # -----------------------------------------------
    # Save the updated DataFrame back to CSV
    # -----------------------------------------------
    df.to_csv(team_path, index=False)

# Get all CSV files in the 'teams' directory
team_files = [f for f in os.listdir(teams_dir) if f.endswith('.csv')]

# Iterate through each team file and process it
for team_file in team_files:
    team_path = os.path.join(teams_dir, team_file)
    print(f'Processing {team_file}...')
    try:
        process_team_file(team_path)
    except Exception as e:
        print(f'Error processing {team_file}: {e}')

print('All team files have been processed successfully.')