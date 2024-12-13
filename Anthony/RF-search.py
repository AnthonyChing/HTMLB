from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
from joblib import Parallel, delayed

# Load data
df = pd.read_csv('../uranus/preprocessing/undropped_train.csv')

# Fill categorical columns with "Unknown"
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

n_estimators = 10000
max_features = ['sqrt', 'log2']
max_depth = [5, 10, 15]
min_samples_split = [2, 5, 10]
min_samples_leaf = [1, 2, 4]
params_list = [{"n_estimators": n_estimators, 
                "max_features": features, 
                "max_depth": depth, 
                "min_samples_split": samples_split, 
                "min_samples_leaf": samples_leaf}
                for features in max_features 
                for depth in max_depth 
                for samples_split in min_samples_split
                for samples_leaf in min_samples_leaf]

def calc(params):
    rf = RandomForestClassifier(n_estimators=params["n_estimators"], 
                                max_features=params["max_features"],
                                max_depth=params["max_depth"],
                                min_samples_split=params["min_samples_split"],
                                min_samples_leaf=params["min_samples_leaf"],
                                random_state=1126)

    # Train the model
    rf.fit(params["X_train"], params["y_train"])

    # Predict and evaluate
    y_pred = rf.predict(params["X_val"])

    print(params["n_estimators"],
          params["max_features"], 
          params["max_depth"], 
          params["min_samples_split"], 
          params["min_samples_leaf"],
          accuracy_score(params["y_val"], y_pred))
    
    return(params["n_estimators"],
          params["max_features"], 
          params["max_depth"], 
          params["min_samples_split"], 
          params["min_samples_leaf"],
          accuracy_score(params["y_val"], y_pred))

for param in params_list:
    param["X_train"] = X_train_1
    param["y_train"] = y_train_1
    param["X_val"] = X_val_1
    param["y_val"] = y_val_1
results = Parallel(n_jobs=-1)(delayed(calc)(params) for params in params_list)

f = open(f'RF-search-late-stage1-grid.txt', 'w')
sorted_results = sorted(
    params_list,
    key=lambda x: (
        x["n_estimators"],
        x["max_features"],
        x["max_depth"],
        x["min_samples_split"],
        x["min_samples_leaf"]
    )
)
for accuracy in sorted_results:
    f.write(str(accuracy) + "\n")
max_result = max(sorted_results, key=lambda x: x[5])
f.write(max_result+"\n")
print(max_result)
f.close()

for param in params_list:
    param["X_train"] = X_train_2
    param["y_train"] = y_train_2
    param["X_val"] = X_val_2
    param["y_val"] = y_val_2
results = Parallel(n_jobs=-1)(delayed(calc)(params) for params in params_list)

f = open(f'RF-search-late-stage2-grid.txt', 'w')
sorted_results = sorted(
    params_list,
    key=lambda x: (
        x["n_estimators"],
        x["max_features"],
        x["max_depth"],
        x["min_samples_split"],
        x["min_samples_leaf"]
    )
)
for accuracy in sorted_results:
    f.write(str(accuracy) + "\n")
max_result = max(sorted_results, key=lambda x: x[5])
f.write(max_result+"\n")
print(max_result)
f.close()