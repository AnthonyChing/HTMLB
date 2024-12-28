import itertools
import os
import pandas as pd
import numpy as np
import lightgbm as lgb
import pickle

def preprocess_data(input_file='../../preprocessing/undropped_train.csv'):

    # Read the CSV file
    df = pd.read_csv(input_file)

    # Convert True/False entries to 1/0 in the entire DataFrame
    pd.set_option('future.no_silent_downcasting', True)
    df = df.replace({True: 1, False: 0})
    df = df.infer_objects(copy=False)

    # Save the updated DataFrame to a new CSV file or overwrite the existing one
    df.to_csv('final_updated.csv', index=False)  # Change 'final_updated.csv' to 'final.csv' to overwrite

    print("True entries replaced with 1 and False entries replaced with 0.")


def prepare_data(input_file='final_updated.csv', 
                processed_data_dir='processed_data'):
    """
    Loads and preprocesses the dataset, then splits it into training and validation sets.
    The processed datasets are saved to disk for later use.
    """
    try:
        # Step 1: Load dataset
        data = pd.read_csv(input_file)
        print(f"Loaded dataset with {data.shape[0]} records and {data.shape[1]} features.")

        # Step 2: Drop unwanted columns
        drop_columns = [
            'id', 'home_team_abbr', 'away_team_abbr',
            'home_pitcher', 'away_pitcher', 'home_team_season', 'away_team_season'
        ]
        data = data.drop(columns=drop_columns, errors='ignore')
        # for col in drop_columns:
        #     print(f"drop column: {col}")

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
        if 'is_night_game' not in data.columns:
            raise ValueError("Target column 'is_night_game' not found.")
        y = data['is_night_game']
        X = data.drop(['is_night_game'], axis=1)

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

        train_mask = (
            (X['year'] < 2023)
        )
        val_mask = (
            (X['year'] >= 2023)
        )

        train_mask = train_mask & ~val_mask
        X_train, y_train = X[train_mask].copy(), y[train_mask].copy()
        X_val, y_val = X[val_mask].copy(), y[val_mask].copy()

        # Drop date-related columns
        date_related_cols = ['date', 'year', 'month', 'day']
        X_train.drop(columns=date_related_cols, errors='ignore', inplace=True)
        X_val.drop(columns=date_related_cols, errors='ignore', inplace=True)

        # **Important:** Handle categorical features instead of dropping them
        for col in categorical_cols:
            if col in X_train.columns:
                X_train.drop(col, axis=1, inplace=True)
            if col in X_val.columns:
                X_val.drop(col, axis=1, inplace=True)

        return X_train, y_train, X_val, y_val, X, y

    except FileNotFoundError:
        print(f"Error: File {input_file} not found.")
    except ValueError as ve:
        print("ValueError:", ve)
    except Exception as e:
        print("Unexpected error:", e)

def train_model(X_train, y_train, X_late, y_late, X, y, params, model_dir, ein_file):
    """
    Trains a LightGBM model and evaluates it on training and validation sets.
    """
    # Penalize False more
    weight_false = 1.15
    weight_true = 1

    y_weights = np.where(y_train==0, weight_false, weight_true)

    # Create LightGBM datasets
    lgb_train = lgb.Dataset(
        X_train, 
        y_train,
        weight=y_weights
    )
    lgb_late = lgb.Dataset(X_late, y_late, reference=lgb_train)

    # Train the model
    print("Starting model training...")
    gbm = lgb.train(
        params,
        lgb_train,
        valid_sets=[lgb_train, lgb_late],
    )
    print("Model training completed.")

    # **Make Predictions**

    Threshold = 0.5

    # Predict on Training Set
    print("Predicting on training set...")
    y_train_pred_prob = gbm.predict(X_train, num_iteration=gbm.best_iteration)
    y_train_pred = (y_train_pred_prob >= Threshold).astype(int)

    # Calculate E_in Metrics
    ein_train_accuracy = np.mean(y_train_pred == y_train)  # Accuracy calculation
    print(f"E_train_in Accuracy: {ein_train_accuracy:.5f}")

    # Predict on Validation Set
    print("Predicting on validation set...")
    y_late_pred_prob = gbm.predict(X_late, num_iteration=gbm.best_iteration)
    y_late_pred = (y_late_pred_prob >= Threshold).astype(int)

    # Calculate E_in Metrics
    ein_late_accuracy = np.mean(y_late_pred == y_late)  # Accuracy calculation
    print(f"E_late_in Accuracy: {ein_late_accuracy:.5f}")

    with open(ein_file, 'a') as f:
        f.write(f"E_late Accuracy: {ein_late_accuracy:.4f}, Params: {params}\n")

    # **Save the Model as a Pickle File**
    model_filename = "_".join([f"{k}={v}" for k, v in params.items() if k != 'verbose']) + ".pkl"
    model_path_pkl = os.path.join(model_dir, model_filename)
    with open(model_path_pkl, 'wb') as f:
        pickle.dump(gbm, f)
    print(f"Model saved as pickle to '{model_path_pkl}'.")

    return ein_late_accuracy

def main():
    preprocess_data()
    # Prepare Data
    X_train, y_train, X_late, y_late, X, y = prepare_data()
    if X is None:
        return

    # Create directories if they don't exist
    model_dir = 'models'
    os.makedirs(model_dir, exist_ok=True)
    ein_file = 'E_late.txt'

    # Clear or initialize Ein.txt
    with open(ein_file, 'w') as f:
        f.write("E_late Accuracy Log:\n")

    # Define parameter grid
    param_grid = {
        'num_boost_round': [100, 200],
        'learning_rate': [0.001],
        'num_leaves': [15],
        'feature_fraction': [0.6],
        'bagging_fraction': [1],
        'min_child_samples': [100],
        'verbose': [0]
    }

    # Generate all combinations of parameters
    keys, values = zip(*param_grid.items())
    param_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

    # Train and evaluate models
    for params in param_combinations:
        params.update({
            'objective': 'binary',
        })
        train_model(X_train, y_train, X_late, y_late, X, y, params, model_dir, ein_file)

    print("\n--- All models trained ---")


if __name__ == "__main__":
    main()