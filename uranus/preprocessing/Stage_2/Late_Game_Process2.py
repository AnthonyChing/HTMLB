# preprocess.py

import pandas as pd
import numpy as np
import os

def prepare_data(input_file='undropped_train.csv', 
                processed_data_dir='Late_Game_Dataset2'):
    """
    Loads and preprocesses the dataset, then splits it into training and validation sets.
    The processed datasets are saved to disk as CSV files for later use.
    """
    try:
        # Ensure the processed data directory exists
        os.makedirs(processed_data_dir, exist_ok=True)
        
        # Step 1: Load dataset
        data = pd.read_csv(input_file)
        print(f"Loaded dataset with {data.shape[0]} records and {data.shape[1]} features.")

        # # Step 2: Drop unwanted columns
        # drop_columns = [
        #     'id', 'home_team_abbr', 'away_team_abbr', 'is_night_game',
        #     'home_pitcher', 'away_pitcher', 'home_team_season', 'away_team_season'
        # ]
        # data = data.drop(columns=drop_columns, errors='ignore')

        # Step 3: Date handling
        if 'date' in data.columns:
            data['date'] = pd.to_datetime(data['date'], errors='coerce')
            data = data.dropna(subset=['date'])
            data = data.sort_values('date').reset_index(drop=True)
            print("Dataset sorted chronologically by 'date'.")
        else:
            print("No 'date' column found, proceeding without date-based sorting.")
            data['date'] = pd.NaT

        # Step 4: Separate target and features
        if 'home_team_win' not in data.columns:
            raise ValueError("Target column 'home_team_win' not found.")
        y = data['home_team_win']
        X = data.drop(['home_team_win'], axis=1)

        # Step 5: Identify columns
        numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

        print(f"Numerical columns: {len(numerical_cols)}; Categorical columns: {len(categorical_cols)}")

        # Extract year, month, day
        if 'date' in data.columns:
            X['year'] = data['date'].dt.year
            X['month'] = data['date'].dt.month
            X['day'] = data['date'].dt.day

        # Step 6: Determine train/validation split
        available_years = sorted(X['year'].dropna().unique())
        if 2020 in available_years:
            available_years.remove(2020)

        if len(available_years) == 0:
            raise ValueError("No valid years after excluding 2020.")

        train_mask = (X['year'] < 2023)
        val_mask = (X['year'] >= 2023)

        X_train, y_train = X[train_mask].copy(), y[train_mask].copy()
        X_val, y_val = X[val_mask].copy(), y[val_mask].copy()

        # Drop date-related columns
        # date_related_cols = ['date', 'year', 'month', 'day']
        # X_train.drop(columns=date_related_cols, errors='ignore', inplace=True)
        # X_val.drop(columns=date_related_cols, errors='ignore', inplace=True)

        # Handle categorical features (currently dropping them; consider encoding if needed)
        # for col in categorical_cols:
        #     if col in X_train.columns:
        #         X_train.drop(col, axis=1, inplace=True)
        #     if col in X_val.columns:
        #         X_val.drop(col, axis=1, inplace=True)

        # Save processed datasets as CSV
        X_train.to_csv(os.path.join(processed_data_dir, 'X_train_2.csv'), index=False)
        y_train.to_csv(os.path.join(processed_data_dir, 'y_train_2.csv'), index=False)
        X_val.to_csv(os.path.join(processed_data_dir, 'X_val_2.csv'), index=False)
        y_val.to_csv(os.path.join(processed_data_dir, 'y_val_2.csv'), index=False)

        print(f"Processed data saved to '{processed_data_dir}' directory as CSV files.")

    except FileNotFoundError:
        print(f"Error: File {input_file} not found.")
    except ValueError as ve:
        print("ValueError:", ve)
    except Exception as e:
        print("Unexpected error:", e)

def main():
    prepare_data()

if __name__ == "__main__":
    main()