import os
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import accuracy_score
import pickle
import json

def load_data(train_file, val_file, drop_file='../../feature_importance_LightGBM/drop100.txt'):
    """
    Loads training and validation data from specified CSV files.
    Removes all categorical columns, 'id', and columns specified in drop_file from both datasets.
    """
    try:
        # Load training and validation data
        train_data = pd.read_csv(train_file)
        val_data = pd.read_csv(val_file)
        
        # Load columns to drop from the drop file
        with open(drop_file, 'r') as f:
            drop_columns = [line.strip() for line in f.readlines()]

        # Identify categorical columns
        categorical_cols = train_data.select_dtypes(include=['object', 'category']).columns.tolist()

        # Combine columns to drop
        columns_to_drop = categorical_cols + drop_columns + ['id']

        #print(f"drop: {columns_to_drop}")

        # Clean the datasets
        train_data_clean = train_data.drop(columns=columns_to_drop, errors='ignore')
        val_data_clean = val_data.drop(columns=columns_to_drop, errors='ignore')

        required_cols = ['home_team_win']
        for col in required_cols:
            if col not in train_data_clean.columns or col not in val_data_clean.columns:
                raise ValueError(f"Required column '{col}' not found in one of the datasets.")
        
        X_train = train_data_clean.drop(['home_team_win'], axis=1)
        y_train = train_data_clean['home_team_win']
        
        X_val = val_data_clean.drop(['home_team_win'], axis=1)
        y_val = val_data_clean['home_team_win']
        
        return X_train, y_train, X_val, y_val

    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None, None, None


def train_and_evaluate(X_train, y_train, X_val, y_val, params, model_name):
    """
    Trains a LightGBM model and evaluates it. Saves the model and logs results.
    """
    # Convert training data to LightGBM Dataset
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

    # Train the model
    print("Starting model training...")
    gbm = lgb.train(
        params,
        train_data,
        valid_sets=[train_data, val_data],
    )
    print("Model training completed.")
    
    # Predict probabilities
    y_val_pred = gbm.predict(X_val)
    y_train_pred = gbm.predict(X_train)
    
    # Convert probabilities to binary predictions
    y_val_pred_binary = (y_val_pred > 0.5).astype(int)
    y_train_pred_binary = (y_train_pred > 0.5).astype(int)
    
    # Calculate accuracy
    E_val = accuracy_score(y_val, y_val_pred_binary)
    E_in = accuracy_score(y_train, y_train_pred_binary)
    
    # Save model to tscv_models directory
    os.makedirs("tscv_models", exist_ok=True)
    model_path = os.path.join("tscv_models", f"{model_name}.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(gbm, f)
    
    return E_val, E_in, gbm

def log_results(params, E_val, E_in, fold, file_name="Error.txt"):
    """
    Logs results to the specified file.
    """
    with open(file_name, "a") as f:
        f.write(f"Fold: {fold}, Parameters: {params}, E_val: {E_val:.4f}, E_in: {E_in:.4f}\n")

def get_param_combinations():
    """
    Generates a list of parameter combinations for LightGBM.
    """
    param_grid = {
        'num_boost_round': [300],
        'learning_rate': [0.001],
        'num_leaves': [15],
        'feature_fraction': [0.6],
        'bagging_fraction': [1],
        'min_child_samples': [100],
        'verbose': [0]
    }
    
    from itertools import product
    keys, values = zip(*param_grid.items())
    return [dict(zip(keys, v)) for v in product(*values)]

def main():
    test_files = [
        '../../../Anthony/train_data_tscv_0.csv',
        '../../../Anthony/train_data_tscv_1.csv',
        '../../../Anthony/train_data_tscv_2.csv',
        '../../../Anthony/train_data_tscv_3.csv'
    ]
    val_files = [
        '../../../Anthony/val_data_tscv_0.csv',
        '../../../Anthony/val_data_tscv_1.csv',
        '../../../Anthony/val_data_tscv_2.csv',
        '../../../Anthony/val_data_tscv_3.csv'
    ]
    
    if len(test_files) != len(val_files):
        print("Error: The number of test files and validation files do not match.")
        return
    
    param_combinations = get_param_combinations()
    print(f"Testing {len(param_combinations)} parameter combinations...")
    
    
    for params in param_combinations:
        print(f"\nTesting Parameters: {params}")
        total_E_val = 0  # Sum of validation accuracies
        total_E_in = 0   # Sum of training accuracies
        total_weight = 10  # Sum of weights
        weights = [1, 2, 3, 4]  # Weights for each fold

        for i, (test_file, val_file) in enumerate(zip(test_files, val_files)):
            print(f"Processing Fold {i}...")
            
            X_train, y_train, X_val, y_val = load_data(test_file, val_file)
            if X_train is None:
                print(f"Skipping Fold {i} due to data loading issues.")
                continue
            
            model_name = "_".join([f"{k}={v}" for k, v in params.items() if k != 'verbose']) + ".pkl"
            E_val, E_in, _ = train_and_evaluate(X_train, y_train, X_val, y_val, params, model_name)
            
            # Accumulate weighted accuracies
            total_E_val += weights[i] * E_val
            total_E_in += weights[i] * E_in

        # Compute overall weighted accuracies
        E_val = total_E_val / total_weight
        E_in = total_E_in / total_weight

        print(f"Weighted Validation Accuracy (Eval) for parameters {params}: {E_val:.4f}")
        print(f"Weighted Training Accuracy (Ein) for parameters {params}: {E_in:.4f}")
        
        # Save the overall results to a log file
        with open("Results.txt", "a") as f:
            f.write(f"Parameters: {params}, Eval: {E_val:.4f}, Ein: {E_in:.4f}\n")
        
    
    
    # final_train_file = 'final_train.csv'
    # train_final_model(final_train_file, best_params)

if __name__ == "__main__":
    main()