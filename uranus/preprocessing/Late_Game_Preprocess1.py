# preprocess.py

import pandas as pd
import numpy as np
import os

def prepare_data(input_file='undropped_train.csv', 
                processed_data_dir='Late_Game_Dataset1'):
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

        # Step 2: Date handling
        if 'date' in data.columns:
            data['date'] = pd.to_datetime(data['date'], errors='coerce')
            data = data.dropna(subset=['date'])
            data = data.sort_values('date').reset_index(drop=True)
            print("Dataset sorted chronologically by 'date'.")
        else:
            print("No 'date' column found, proceeding without date-based sorting.")
            data['date'] = pd.NaT

        X = data

        # Extract year, month, day
        if 'date' in data.columns:
            X['year'] = data['date'].dt.year
            X['month'] = data['date'].dt.month
            X['day'] = data['date'].dt.day

        train_mask = (
            ((X['month'] >= 4) & (X['month'] <= 6)) | 
            ((X['month'] == 7) & (X['day'] <= 15))
        )
        val_mask = (
            (X['month'] == 7) &
            (X['day'] > 15)
        )

        X_train = X[train_mask].copy()
        X_val = X[val_mask].copy()

        # Save processed datasets as CSV
        X_train.to_csv(os.path.join(processed_data_dir, 'train_data_1.csv'), index=False)
        X_val.to_csv(os.path.join(processed_data_dir, 'val_data_1.csv'), index=False)

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