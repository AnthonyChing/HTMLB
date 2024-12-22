from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
from joblib import Parallel, delayed

# Load data
df = pd.read_csv('../uranus/preprocessing/undropped_train.csv')

categorical_columns = df.select_dtypes(include=['object']).columns
categorical_columns = categorical_columns.drop('is_night_game')

# Encode categorical columns (factorize to numeric codes)
encodings = {}
for col in categorical_columns:
    df[col], encodings[col] = pd.factorize(df[col])

train_data_1 = pd.read_csv('../uranus/preprocessing/Late_Game_Dataset1/train_data_1.csv')
val_data_1 = pd.read_csv('../uranus/preprocessing/Late_Game_Dataset1/val_data_1.csv')

# Apply the same encoding to test data
for col in categorical_columns:
    train_data_1[col] = train_data_1[col].map({value: idx for idx, value in enumerate(encodings[col])})
    val_data_1[col] = val_data_1[col].map({value: idx for idx, value in enumerate(encodings[col])})

X_train_1 = train_data_1.drop(columns=['home_team_win', 'date', 'id']).values  # Features
y_train_1 = train_data_1['home_team_win'].values  # Target

X_val_1 = val_data_1.drop(columns=['home_team_win', 'date', 'id']).values  # Features
y_val_1 = val_data_1['home_team_win'].values  # Target

train_data_2 = pd.read_csv('../uranus/preprocessing/Late_Game_Dataset2/train_data_2.csv')
val_data_2 = pd.read_csv('../uranus/preprocessing/Late_Game_Dataset2/val_data_2.csv')

# Apply the same encoding to test data
for col in categorical_columns:
    train_data_2[col] = train_data_2[col].map({value: idx for idx, value in enumerate(encodings[col])})
    val_data_2[col] = val_data_2[col].map({value: idx for idx, value in enumerate(encodings[col])})

X_train_2 = train_data_2.drop(columns=['home_team_win', 'date', 'id']).values  # Features
y_train_2 = train_data_2['home_team_win'].values  # Target

X_val_2 = val_data_2.drop(columns=['home_team_win', 'date', 'id']).values  # Features
y_val_2 = val_data_2['home_team_win'].values  # Target

n_estimators = [100 * i for i in range(1, 11, 2)]
max_features = ['sqrt', 'log2', None]
max_depth = [10, 15, None]
params_list = [{"n_estimators": estimators, 
                "max_features": features, 
                "max_depth": depth}
                for estimators in n_estimators 
                for features in max_features 
                for depth in max_depth]

def toString(params):
    return (str(params["n_estimators"]) + " " +
            str(params["max_features"]) + " " +
            str(params["max_depth"]) + " ")

def calc(params):
    rf = RandomForestClassifier(n_estimators=params["n_estimators"], 
                                max_features=params["max_features"],
                                max_depth=params["max_depth"],
                                oob_score=True,
                                random_state=1126)

    # Train the model
    rf.fit(params["X_train"], params["y_train"])

    # Predict and evaluate
    y_pred = rf.predict(params["X_val"])
    accuracy = accuracy_score(params["y_val"], y_pred)

    # Add accuracy to params dictionary
    params["accuracy"] = accuracy

    print(toString(params) + str(accuracy))
    
    return params

for param in params_list:
    param["X_train"] = X_train_1
    param["y_train"] = y_train_1
    param["X_val"] = X_val_1
    param["y_val"] = y_val_1
results = Parallel(n_jobs=-1)(delayed(calc)(params) for params in params_list)

max_features_priority = {"sqrt": 1, "log2": 2, None: 3}
with open(f'RF-search-late-stage1-grid-less.txt', 'w') as f:
    sorted_results = sorted(
        results,
        key=lambda x: (
            x["n_estimators"],
            max_features_priority.get(x["max_features"], float('inf')),
            x["max_depth"] if x["max_depth"] is not None else 0
        )
    )
    for result in sorted_results:
        f.write(toString(result) + " " + str(result["accuracy"]) + "\n")
    max_result = max(sorted_results, key=lambda x: x["accuracy"])
    f.write(toString(max_result) + " " + str(max_result["accuracy"]) + "\n")
    print("Stage 1 Optimal: " + toString(max_result) + " " + str(max_result["accuracy"]))
# Stage 1 Optimal: 700 sqrt 15  0.5606166783461808
for param in params_list:
    param["X_train"] = X_train_2
    param["y_train"] = y_train_2
    param["X_val"] = X_val_2
    param["y_val"] = y_val_2
results = Parallel(n_jobs=-1)(delayed(calc)(params) for params in params_list)

with open(f'RF-search-late-stage2-grid-less.txt', 'w') as f:
    sorted_results = sorted(
        results,
        key=lambda x: (
            x["n_estimators"],
            max_features_priority.get(x["max_features"], float('inf')),
            x["max_depth"] if x["max_depth"] is not None else 0
        )
    )
    for result in sorted_results:
        f.write(toString(result) + " " + str(result["accuracy"]) + "\n")
    max_result = max(sorted_results, key=lambda x: x["accuracy"])
    f.write(toString(max_result) + " " + str(max_result["accuracy"]) + "\n")
    print("Stage 2 Optimal: " + toString(max_result) + " " + str(max_result["accuracy"]))
# Stage 2 Optimal: 300 log2 10  0.5597996242955542