import pandas as pd
import re
import sys
import argparse
import os

# Function to clean whitespace in string columns
def clean_whitespace(value):
    if isinstance(value, str):
        # Remove leading and trailing spaces
        value = value.strip()
        # Replace multiple spaces with a single space
        value = re.sub(r'\s+', ' ', value)
        return value
    return value

def separate_home_teams(df, output_dir):
    """
    Separates the DataFrame into individual home team CSV files.

    Each CSV file will contain all games where the team is the home team.
    """
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nOutput directory '{output_dir}' is ready.")

    # Identify all unique home teams
    home_teams = df['home_team_abbr'].unique()
    print(f"Total unique home teams found: {len(home_teams)}")

    for team in home_teams:
        # Filter rows where the team is the home team
        team_games = df[df['home_team_abbr'] == team]

        # Define the output file path
        output_file = os.path.join(output_dir, f"{team}_home_games.csv")

        # Save to CSV
        try:
            team_games.to_csv(output_file, index=False)
            print(f"Saved home games for team '{team}' to '{output_file}'.")
        except Exception as e:
            print(f"Error saving file for team '{team}': {e}")

def main():
    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(description='Clean, sort, and separate a CSV file by home team.')
    parser.add_argument('input_file', help='Path to the input CSV file.')
    parser.add_argument('output_file', help='Path to save the sorted and cleaned CSV file.')
    parser.add_argument('--delimiter', default=',', help='Delimiter used in the CSV file (default is comma).')
    args = parser.parse_args()

    input_file = args.input_file
    output_file = args.output_file
    delimiter = args.delimiter

    # Load the CSV file with error handling
    try:
        df = pd.read_csv(input_file, delimiter=delimiter)
        print(f"Successfully loaded '{input_file}'.")
    except FileNotFoundError:
        print(f"Error: The file '{input_file}' was not found.")
        sys.exit(1)
    except pd.errors.EmptyDataError:
        print(f"Error: The file '{input_file}' is empty.")
        sys.exit(1)
    except pd.errors.ParserError as e:
        print(f"Error parsing '{input_file}': {e}")
        sys.exit(1)

    # Print the original column names for debugging
    print("\nOriginal Column Names:")
    for col in df.columns:
        print(f"'{col}'")

    # Clean column names: strip spaces and convert to lowercase
    df.columns = df.columns.str.strip().str.lower()
    print("\nCleaned Column Names:")
    for col in df.columns:
        print(f"'{col}'")

    # Define the expected sort columns
    expected_sort_columns = [
        'date',
        'home_team_abbr',
        'away_team_abbr',
        'home_pitcher',
        'away_pitcher'
    ]

    # Check if expected columns are present
    missing_columns = [col for col in expected_sort_columns if col not in df.columns]
    if missing_columns:
        print(f"\nError: The following sort columns are missing from the CSV: {missing_columns}")
        print("Please verify the column names and ensure they match exactly (case-insensitive and no extra spaces).")
        sys.exit(1)
    else:
        print("\nAll required sort columns are present.")

    # Strip redundant spaces across all string columns using apply + map
    object_columns = df.select_dtypes(include=['object']).columns
    df[object_columns] = df[object_columns].apply(lambda col: col.map(clean_whitespace))
    print("Redundant spaces have been stripped from all string fields.")

    # Optional: Standardize team abbreviations to uppercase (helps in consistency)
    df['home_team_abbr'] = df['home_team_abbr'].str.upper()
    df['away_team_abbr'] = df['away_team_abbr'].str.upper()
    print("Team abbreviations have been standardized to uppercase.")

    # Convert the 'date' column to datetime format
    try:
        df['date'] = pd.to_datetime(df['date'])
        print("'date' column has been converted to datetime format.")
    except Exception as e:
        print(f"Error converting 'date' column to datetime: {e}")
        sys.exit(1)

    # Define sorting order
    sort_columns = expected_sort_columns

    # Sort the DataFrame
    try:
        sorted_df = df.sort_values(by=sort_columns)
        print("Data has been sorted based on the specified columns.")
    except Exception as e:
        print(f"Error during sorting: {e}")
        sys.exit(1)

    # Save the sorted and cleaned DataFrame to a new CSV file
    try:
        sorted_df.to_csv(output_file, index=False)
        print(f"Sorted and cleaned CSV file has been saved as '{output_file}'.")
    except Exception as e:
        print(f"Error saving the sorted CSV file: {e}")
        sys.exit(1)

    # Separate each home team into different files within the 'sort' directory
    output_dir = 'sort'
    print(f"\nSeparating each home team into different files in the '{output_dir}' directory...")
    separate_home_teams(sorted_df, output_dir)
    print(f"All team-specific CSV files have been saved in the '{output_dir}' directory.")

if __name__ == "__main__":
    main()