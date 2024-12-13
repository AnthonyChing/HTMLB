import os
import pandas as pd

# Define the directory containing team CSV files
teams_dir = 'away_teams'

# List of columns to fill by averaging the previous 1 game and next 1 game
rolling_columns = [
    'away_batting_batting_avg_10RA',
    'away_batting_onbase_perc_10RA',
    'away_batting_onbase_plus_slugging_10RA',
    'away_batting_leverage_index_avg_10RA',
    'away_batting_RBI_10RA',
    'away_pitching_earned_run_avg_10RA',
    'away_pitching_SO_batters_faced_10RA',
    'away_pitching_H_batters_faced_10RA',
    'away_pitching_BB_batters_faced_10RA',
    'away_pitcher_earned_run_avg_10RA',
    'away_pitcher_SO_batters_faced_10RA',
    'away_pitcher_H_batters_faced_10RA',
    'away_pitcher_BB_batters_faced_10RA',
    'away_team_errors_mean',
    'away_team_errors_std',
    'away_team_errors_skew',
    'away_team_spread_mean',
    'away_team_spread_std',
    'away_team_spread_skew',
    'away_team_wins_mean',
    'away_team_wins_std',
    'away_team_wins_skew',
    'away_batting_batting_avg_mean',
    'away_batting_batting_avg_std',
    'away_batting_batting_avg_skew',
    'away_batting_onbase_perc_mean',
    'away_batting_onbase_perc_std',
    'away_batting_onbase_perc_skew',
    'away_batting_onbase_plus_slugging_mean',
    'away_batting_onbase_plus_slugging_std',
    'away_batting_onbase_plus_slugging_skew',
    'away_batting_leverage_index_avg_mean',
    'away_batting_leverage_index_avg_std',
    'away_batting_leverage_index_avg_skew',
    'away_batting_wpa_bat_mean',
    'away_batting_wpa_bat_std',
    'away_batting_wpa_bat_skew',
    'away_batting_RBI_mean',
    'away_batting_RBI_std',
    'away_batting_RBI_skew',
    'away_pitching_earned_run_avg_mean',
    'away_pitching_earned_run_avg_std',
    'away_pitching_earned_run_avg_skew',
    'away_pitching_SO_batters_faced_mean',
    'away_pitching_SO_batters_faced_std',
    'away_pitching_SO_batters_faced_skew',
    'away_pitching_H_batters_faced_mean',
    'away_pitching_H_batters_faced_std',
    'away_pitching_H_batters_faced_skew',
    'away_pitching_BB_batters_faced_mean',
    'away_pitching_BB_batters_faced_std',
    'away_pitching_BB_batters_faced_skew',
    'away_pitching_leverage_index_avg_mean',
    'away_pitching_leverage_index_avg_std',
    'away_pitching_leverage_index_avg_skew',
    'away_pitching_wpa_def_mean',
    'away_pitching_wpa_def_std',
    'away_pitching_wpa_def_skew',
    'away_pitcher_earned_run_avg_mean',
    'away_pitcher_earned_run_avg_std',
    'away_pitcher_earned_run_avg_skew',
    'away_pitcher_SO_batters_faced_mean',
    'away_pitcher_SO_batters_faced_std',
    'away_pitcher_SO_batters_faced_skew',
    'away_pitcher_H_batters_faced_mean',
    'away_pitcher_H_batters_faced_std',
    'away_pitcher_H_batters_faced_skew',
    'away_pitcher_BB_batters_faced_mean',
    'away_pitcher_BB_batters_faced_std',
    'away_pitcher_BB_batters_faced_skew',
    'away_pitcher_leverage_index_avg_mean',
    'away_pitcher_leverage_index_avg_std',
    'away_pitcher_leverage_index_avg_skew',
    'away_pitcher_wpa_def_mean',
    'away_pitcher_wpa_def_std',
    'away_pitcher_wpa_def_skew'
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
    # Compute away_team_rest
    # -----------------------

    # Ensure 'away_team_rest' column exists
    if 'away_team_rest' not in df.columns:
        raise KeyError(f"'away_team_rest' column not found in {team_path}")

    # Calculate difference in days between current game and previous game
    df['prev_date'] = df['date'].shift(1)
    df['diff_days'] = (df['date'] - df['prev_date']).dt.days

    # For the first game, fill NaN with 1
    df['diff_days'] = df['diff_days'].fillna(1)

    # Apply the rule to NaN: if diff_days >= 160, set rest to 1.0, else to the actual difference
    df['away_team_rest'] = df['away_team_rest'].fillna(
        df['diff_days'].apply(lambda x: 1.0 if x >= 160 else float(x))
    )

    # Drop temporary columns
    df.drop(columns=['prev_date', 'diff_days'], inplace=True)

    # -------------------------
    # Compute away_pitcher_rest
    # -------------------------

    # Ensure 'away_pitcher' column exists
    if 'away_pitcher' not in df.columns:
        raise KeyError(f"'away_pitcher' column not found in {team_path}")

    # Group by 'away_pitcher' and calculate difference in days between appearances
    df['prev_pitcher_date'] = df.groupby('away_pitcher')['date'].shift(1)
    df['pitcher_diff_days'] = (df['date'] - df['prev_pitcher_date']).dt.days

    # Fill NaN (first appearance) with 1
    df['pitcher_diff_days'] = df['pitcher_diff_days'].fillna(1)

    # Apply the rule: if pitcher_diff_days >=60, set rest to 1.0, else to the actual difference
    df['away_pitcher_rest'] = df['pitcher_diff_days'].apply(lambda x: 1.0 if x >= 60 else float(x))

    # Apply the rule to NaN: if pitcher_diff_days >=60, set rest to 1.0, else to the actual difference
    df['away_pitcher_rest'] = df['away_pitcher_rest'].fillna(
        df['pitcher_diff_days'].apply(lambda x: 1.0 if x >= 60 else float(x))
    )

    # Drop temporary columns
    df.drop(columns=['prev_pitcher_date', 'pitcher_diff_days'], inplace=True)

    # -----------------------------------------------
    # Compute Rolling Averages for Specified Columns
    # -----------------------------------------------
    # Ensure rolling columns are numeric
    for col in rolling_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    #First pass: fill isolated NaNs
    group_keys = ['season']
    for col in rolling_columns:
        df[col] = df.groupby(group_keys)[col].transform(
            lambda x: x.fillna((x.shift(1) + x.shift(-1)) / 2)
        )

    # Second pass: forward fill remaining NaNs
    for col in rolling_columns:
        df[col] = df.groupby(group_keys)[col].ffill()

    # Third pass: backward fill if any NaNs still remain
    for col in rolling_columns:
        df[col] = df.groupby(group_keys)[col].bfill()

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