import os
import pandas as pd
import joblib
import numpy as np

def remove_columns(data_file='../../../stage 2/2024_test_data.csv', drop_file='../../feature_importance_LightGBM/drop100.txt'):
    """
    Loads data from a CSV file and removes specified columns from it.
    
    Parameters:
    - data_file: Path to the CSV file from which to remove columns.
    - drop_file: Path to the file containing the names of the columns to drop.
    
    Returns:
    - A DataFrame with the specified columns removed.
    """
    try:
        # Load the data from the CSV file
        data = pd.read_csv(data_file)

        # Read the columns to drop from the drop file
        with open(drop_file, 'r') as f:
            drop_columns = [line.strip() for line in f.readlines()]

        # Remove the specified columns
        cleaned_data = data.drop(columns=drop_columns, errors='ignore')

        # Optionally, save the cleaned data to a new CSV file
        if cleaned_data is not None:
            cleaned_data.to_csv('cleaned_2024_test_data.csv', index=False)
            print("Cleaned data saved as 'cleaned_2024_test_data.csv'.")

        return None
    except Exception as e:
        print(f"Error processing data: {e}")
        return None

def main():
    remove_columns()
    # Directory containing the model files
    model_directory = './D\'_models'
    predict_path = 'cleaned_2024_test_data.csv'
    true_labels_file = '../../../stage 2/2024_test_label.csv'
    eout_file = 'Eout.txt'

    # Get all .pkl files in the directory
    model_files = [f for f in os.listdir(model_directory) if f.endswith('.pkl')]

    if not model_files:
        print(f"No .pkl files found in directory '{model_directory}'.")
        return

    try:
        # Load prediction data
        X_pred = pd.read_csv(predict_path)
        print(f"Successfully loaded prediction data from '{predict_path}'.")
        
        # Drop categorical columns
        X_pred = X_pred.select_dtypes(include=['number'])
        print("Dropped categorical columns. Only numerical data will be used for predictions.")
        X_pred.drop(['id'], axis=1, inplace=True)
    except Exception as e:
        print(f"Error loading prediction data: {e}")
        return

    with open(eout_file, 'w') as f:
        f.write("Model,E_out\n")  # Header for the output file

        for model_file in model_files:
            model_path = os.path.join(model_directory, model_file)
            try:
                # Load the model
                model = joblib.load(model_path)
                print(f"Successfully loaded the model from '{model_path}'.")
                
                # Generate predictions
                preds = model.predict(X_pred)
                Threshold = 0.5
                if preds.dtype != 'int':
                    preds = (preds >= Threshold).astype(int)
                preds_bool = preds.astype(bool)

                # Save predictions temporarily
                predictions_file = 'predictions_temp.csv'
                output = pd.DataFrame({
                    'id': X_pred.index,
                    'home_team_win': preds_bool
                })
                output.to_csv(predictions_file, index=False)

                # Calculate E_out
                e_out_accuracy = calculate_e_out(predictions_file, true_labels_file)
                print(f"E_out for {model_file}: {e_out_accuracy:.5f}")
                f.write(f"{e_out_accuracy:.5f},{model_file}\n")
            except Exception as e:
                print(f"Error processing model '{model_file}': {e}")

def calculate_e_out(predictions_file, true_labels_file):
    """
    Calculates the E_out (accuracy on the test dataset) by comparing predictions with true labels.
    """
    try:
        # Load predictions and true labels
        predictions = pd.read_csv(predictions_file)
        true_labels = pd.read_csv(true_labels_file)

        # Ensure both files have the expected format
        if 'id' not in predictions.columns or 'home_team_win' not in predictions.columns:
            raise ValueError("Expected columns 'id' and 'home_team_win' in predictions file.")
        if 'id' not in true_labels.columns or 'home_team_win' not in true_labels.columns:
            raise ValueError("Expected columns 'id' and 'home_team_win' in true labels file.")

        # Align predictions and true labels by 'id'
        merged = pd.merge(predictions, true_labels, on='id', suffixes=('_pred', '_true'))

        # Extract predicted and true labels
        y_pred = merged['home_team_win_pred'].astype(bool)
        y_true = merged['home_team_win_true'].astype(bool)

        # Calculate E_out (accuracy)
        e_out_accuracy = np.mean(y_pred != y_true)
        return e_out_accuracy
    except Exception as e:
        print(f"Error calculating E_out: {e}")
        return 0

if __name__ == '__main__':
    main()