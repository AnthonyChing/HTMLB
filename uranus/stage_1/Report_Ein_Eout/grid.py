import itertools
import os
import pandas as pd
from sklearn.metrics import (
    log_loss,
    roc_auc_score,
    f1_score
)
import numpy as np
import lightgbm as lgb
import pickle

def prepare_data(input_file='../../preprocessing/undropped_train.csv'):
    """
    Loads and preprocesses the dataset.
    """
    try:
        # Step 1: Load dataset
        data = pd.read_csv(input_file)
        print(f"Loaded dataset with {data.shape[0]} records and {data.shape[1]} features.")

        # Step 2: Drop unwanted columns
        drop_columns = [
            'id', 'home_team_abbr', 'away_team_abbr', 'is_night_game',
            'home_pitcher', 'away_pitcher', 'home_team_season', 'away_team_season'
        ]
        data = data.drop(columns=drop_columns, errors='ignore')

        # Step 4: Separate target and features
        if 'home_team_win' not in data.columns:
            raise ValueError("Target column 'home_team_win' not found.")
        y = data['home_team_win']
        X = data.drop(['home_team_win'], axis=1)

        # Step 5: Identify columns
        numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

        print(f"Numerical columns: {len(numerical_cols)}; Categorical columns: {len(categorical_cols)}")

        return X, y
    except FileNotFoundError:
        print(f"Error: File {input_file} not found.")
    except ValueError as ve:
        print("ValueError:", ve)
    except Exception as e:
        print("Unexpected error:", e)

def train_model(X, y, params, model_dir, ein_file):
    """
    Trains a LightGBM model, saves it, and logs E_in accuracy.
    """
    # Before training, drop date-related columns from X
    X_full = X.copy()
    if 'date' in X_full.columns:
        X_full = X_full.drop(columns=['date', 'year', 'month', 'day'], errors='ignore')

    # Prepare dataset with weights
    weight_false = 1.15
    weight_true = 1
    weights = np.where(y == 0, weight_false, weight_true)
    lgb_train_full = lgb.Dataset(X_full, y, weight=weights)

    print(f"Starting model training with parameters:\n{params}")
    gbm = lgb.train(
        params,
        lgb_train_full,
        valid_sets=[lgb_train_full]
    )
    print("Model training completed.")

    # Predict on Full Training Data to Calculate E_in
    y_full_pred_prob = gbm.predict(X_full, num_iteration=gbm.best_iteration)
    y_full_pred = (y_full_pred_prob >= 0.5).astype(int)

    # Calculate E_in Metrics
    ein_accuracy = np.mean(y_full_pred != y)  # Accuracy calculation
    print(f"E_in Accuracy: {ein_accuracy:.4f}")

    # **Save the Model as a Pickle File**
    model_filename = "_".join([f"{k}={v}" for k, v in params.items() if k != 'verbose']) + ".pkl"
    model_path_pkl = os.path.join(model_dir, model_filename)
    with open(model_path_pkl, 'wb') as f:
        pickle.dump(gbm, f)
    print(f"Model saved as pickle to '{model_path_pkl}'.")

    # Append E_in accuracy and parameters to Ein.txt
    with open(ein_file, 'a') as f:
        f.write(f"Params: {params}, E_in Accuracy: {ein_accuracy:.4f}\n")

    return ein_accuracy

def main():
    # Prepare Data
    X, y = prepare_data()
    if X is None:
        return

    # Create directories if they don't exist
    model_dir = 'models'
    os.makedirs(model_dir, exist_ok=True)
    ein_file = 'Ein.txt'

    # Clear or initialize Ein.txt
    with open(ein_file, 'w') as f:
        f.write("E_in Accuracy Log:\n")

    # Define parameter grid
    param_grid = {
        'n_estimators':[100, 200, 300],
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
        train_model(X, y, params, model_dir, ein_file)

    print("\n--- All models trained and logged to 'Ein.txt' ---")


if __name__ == "__main__":
    main()