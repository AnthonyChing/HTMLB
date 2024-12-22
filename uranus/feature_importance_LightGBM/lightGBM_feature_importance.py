import pandas as pd
import numpy as np
import lightgbm as lgb

def prepare_data(input_file='final_train.csv'):
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
            'id', 'home_team_abbr', 'away_team_abbr', 'is_night_game',
            'home_pitcher', 'away_pitcher', 'home_team_season', 'away_team_season'
        ]
        data = data.drop(columns=drop_columns, errors='ignore')

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

        # Identify and remove datetime columns
        datetime_cols = X.select_dtypes(include=['datetime64[ns]']).columns
        if not datetime_cols.empty:
            print(f"The following datetime columns are removed: {datetime_cols.tolist()}")
        X = X.drop(columns=datetime_cols)

        # Step 5: Identify columns
        numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

        print(f"Numerical columns: {len(numerical_cols)}; Categorical columns: {len(categorical_cols)}")

        return  X, y

    except FileNotFoundError:
        print(f"Error: File {input_file} not found.")
    except ValueError as ve:
        print("ValueError:", ve)
    except Exception as e:
        print("Unexpected error:", e)

def save_feature_importance(model, feature_names):
    """
    Retrieves and outputs feature importances of a trained LightGBM model in descending order.
    """
    # Get feature importances
    importances = model.feature_importance(importance_type='split')
    
    # Create a DataFrame to organize feature importance data
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)

    # Save to file
    with open('features_importance_split.txt', 'w') as f:
        f.write("Feature Importances (sorted by importance):\n\n")
        for index, row in importance_df.iterrows():
            f.write(f"{row['Feature']}: {row['Importance']}\n")
    
    return importance_df

def train_model(X, y):
    """
    Trains a LightGBM model and evaluates it on training and validation sets.
    """
    # Penalize False more
    weight_false = 1.15
    weight_true = 1

    weights = np.where(y==0, weight_false, weight_true)

    # Create LightGBM datasets
    lgb_train = lgb.Dataset(
        X, 
        y,
        weight=weights
    )
    lgb_val = lgb.Dataset(X, y, reference=lgb_train)

    # Define parameters
    params = {
        'learning_rate': 0.001,
        'num_leaves': 15,
        'feature_fraction': 0.65,
        'bagging_fraction': 0.65,
        'bagging_freq': 1,
        'min_child_samples': 100,
        'objective': 'binary',
        'verbose': 0
    }

    # Train the model
    print("Starting model training...")
    gbm = lgb.train(
        params,
        lgb_train,
        num_boost_round=1000,
        valid_sets=[lgb_train, lgb_val],
    )
    print("Model training completed.")

    # Save the model
    model_path = 'model.txt'
    gbm.save_model(model_path)
    print(f"Model saved to '{model_path}'.")

    # Save feature importances
    save_feature_importance(gbm, X.columns)

    print(params)

def main():
    # Prepare Data
    X, y = prepare_data()
    if X is not None:
        # Train Model and Evaluate
        train_model(X, y)

if __name__ == "__main__":
    main()