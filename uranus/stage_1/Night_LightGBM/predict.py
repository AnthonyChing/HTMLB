import os
import pandas as pd
import joblib
import numpy as np

def main():
    # Directory containing the model files
    model_directory = './'
    predict_path = 'final_updated.csv'
    true_labels_file = '../../../Aaron/is_night_game/isnightgame_stage_1_label.csv'
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
        
        # Step 2: Drop unwanted columns
        drop_columns = [
            'is_night_game','id', 'home_team_abbr', 'away_team_abbr', 'date',
            'home_pitcher', 'away_pitcher', 'home_team_season', 'away_team_season'
        ]
        X_pred = X_pred.drop(columns=drop_columns, errors='ignore')
        

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
                    'is_night_game': preds_bool
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
        if 'id' not in predictions.columns or 'is_night_game' not in predictions.columns:
            raise ValueError("Expected columns 'id' and 'is_night_game' in predictions file.")
        if 'id' not in true_labels.columns or 'is_night_game' not in true_labels.columns:
            raise ValueError("Expected columns 'id' and 'is_night_game' in true labels file.")

        # Align predictions and true labels by 'id'
        merged = pd.merge(predictions, true_labels, on='id', suffixes=('_pred', '_true'))

        # Extract predicted and true labels
        y_pred = merged['is_night_game_pred'].astype(bool)
        y_true = merged['is_night_game_true'].astype(bool)

        # Calculate E_out (accuracy)
        e_out_accuracy = np.mean(y_pred == y_true)
        return e_out_accuracy
    except Exception as e:
        print(f"Error calculating E_out: {e}")
        return 0

if __name__ == '__main__':
    main()