import pandas as pd

def drop_columns_from_csv(input_file, output_file, columns_to_drop):
    """
    Drops specified columns from a CSV file and saves the result to a new file.

    Args:
        input_file (str): Path to the input CSV file.
        output_file (str): Path to save the output CSV file.
        columns_to_drop (list): List of column names to drop.

    Returns:
        None
    """
    try:
        # Load the CSV file into a DataFrame
        df = pd.read_csv(input_file)
        print(f"Loaded data with {df.shape[0]} rows and {df.shape[1]} columns.")

        # Drop the specified columns
        df = df.drop(columns=columns_to_drop, errors='ignore')  # 'ignore' skips columns that don't exist
        print(f"Columns after dropping: {df.columns.tolist()}")

        # Save the updated DataFrame to the output file
        df.to_csv(output_file, index=False)
        print(f"Updated CSV file saved to {output_file}.")
    except FileNotFoundError:
        print(f"Error: The file '{input_file}' does not exist.")
    except Exception as e:
        print(f"An error occurred: {e}")

# Example usage
if __name__ == "__main__":
    # Input and output file paths
    input_csv = "undropped_train.csv"  # Replace with your input file path
    output_csv = "dropped.csv"  # Replace with your desired output file path

    # Columns to drop
    columns_to_remove = [
            'is_night_game',
            'home_pitcher', 'away_pitcher', 'home_team_season', 'away_team_season'
        ]

    # Call the function
    drop_columns_from_csv(input_csv, output_csv, columns_to_remove)