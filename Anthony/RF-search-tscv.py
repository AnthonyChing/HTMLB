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

def split(train_data, val_data):
    for col in categorical_columns:
        train_data[col] = train_data[col].map({value: idx for idx, value in enumerate(encodings[col])})
        val_data[col] = val_data[col].map({value: idx for idx, value in enumerate(encodings[col])})
    
    X_train = train_data.drop(columns=['home_team_win', 'date', 'id']).values  # Features
    y_train = train_data['home_team_win'].values  # Target

    X_val = val_data.drop(columns=['home_team_win', 'date', 'id']).values  # Features
    y_val = val_data['home_team_win'].values  # Target

    return X_train, y_train, X_val, y_val

train_data_0 = pd.read_csv('train_data_tscv_0.csv')
val_data_0 = pd.read_csv('val_data_tscv_0.csv')
X_train_0, y_train_0, X_val_0, y_val_0 = split(train_data_0, val_data_0)

train_data_1 = pd.read_csv('train_data_tscv_1.csv')
val_data_1 = pd.read_csv('val_data_tscv_1.csv')
X_train_1, y_train_1, X_val_1, y_val_1 = split(train_data_1, val_data_1)

train_data_2 = pd.read_csv('train_data_tscv_2.csv')
val_data_2 = pd.read_csv('val_data_tscv_2.csv')
X_train_2, y_train_2, X_val_2, y_val_2 = split(train_data_2, val_data_2)

train_data_3 = pd.read_csv('train_data_tscv_3.csv')
val_data_3 = pd.read_csv('val_data_tscv_3.csv')
X_train_3, y_train_3, X_val_3, y_val_3 = split(train_data_3, val_data_3)

n_estimators = 10000
max_features = ['sqrt']
max_depth = [10, 15]
min_samples_split = [5, 10]
min_samples_leaf = [2, 4]
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

    rf.fit(X_train_0, y_train_0)
    y_pred = rf.predict(X_val_0)
    acc_0 = accuracy_score(y_val_0, y_pred)
    print("Calculated acc_0: " +
        str(params["n_estimators"]) + " " +
        str(params["max_features"]) + " " +
        str(params["max_depth"]) + " " +
        str(params["min_samples_split"]) + " " +
        str(params["min_samples_leaf"]) + " " +
        str(acc_0))
    
    rf = RandomForestClassifier(n_estimators=params["n_estimators"], 
                                max_features=params["max_features"],
                                max_depth=params["max_depth"],
                                min_samples_split=params["min_samples_split"],
                                min_samples_leaf=params["min_samples_leaf"],
                                random_state=1126)
    
    rf.fit(X_train_1, y_train_1)
    y_pred = rf.predict(X_val_1)
    acc_1 = accuracy_score(y_val_1, y_pred)
    print("Calculated acc_1: " +
        str(params["n_estimators"]) + " " +
        str(params["max_features"]) + " " +
        str(params["max_depth"]) + " " +
        str(params["min_samples_split"]) + " " +
        str(params["min_samples_leaf"]) + " " +
        str(acc_1))
    
    rf = RandomForestClassifier(n_estimators=params["n_estimators"], 
                                max_features=params["max_features"],
                                max_depth=params["max_depth"],
                                min_samples_split=params["min_samples_split"],
                                min_samples_leaf=params["min_samples_leaf"],
                                random_state=1126)
    
    rf.fit(X_train_2, y_train_2)
    y_pred = rf.predict(X_val_2)
    acc_2 = accuracy_score(y_val_2, y_pred)
    print("Calculated acc_2: " +
        str(params["n_estimators"]) + " " +
        str(params["max_features"]) + " " +
        str(params["max_depth"]) + " " +
        str(params["min_samples_split"]) + " " +
        str(params["min_samples_leaf"]) + " " +
        str(acc_2))

    rf = RandomForestClassifier(n_estimators=params["n_estimators"], 
                                max_features=params["max_features"],
                                max_depth=params["max_depth"],
                                min_samples_split=params["min_samples_split"],
                                min_samples_leaf=params["min_samples_leaf"],
                                random_state=1126)
    
    rf.fit(X_train_3, y_train_3)
    y_pred = rf.predict(X_val_3)
    acc_3 = accuracy_score(y_val_3, y_pred)
    print("Calculated acc_3: " +
        str(params["n_estimators"]) + " " +
        str(params["max_features"]) + " " +
        str(params["max_depth"]) + " " +
        str(params["min_samples_split"]) + " " +
        str(params["min_samples_leaf"]) + " " +
        str(acc_3))

    accuracy = (acc_0*1+acc_1*2+acc_2*3+acc_3*4)/10
    params["accuracy"] = accuracy

    print(params["n_estimators"],
          params["max_features"], 
          params["max_depth"], 
          params["min_samples_split"], 
          params["min_samples_leaf"],
          accuracy)
    
    return params

results = Parallel(n_jobs=8)(delayed(calc)(params) for params in params_list)

f = open(f'RF-search-tscv-grid.txt', 'w')
sorted_results = sorted(
    results,
    key=lambda x: (
        x["n_estimators"],
        x["max_features"],
        x["max_depth"],
        x["min_samples_split"],
        x["min_samples_leaf"]
    )
)
for result in sorted_results:
    f.write(str(result["n_estimators"]) + " " +
        str(result["max_features"]) + " " +
        str(result["max_depth"]) + " " +
        str(result["min_samples_split"]) + " " +
        str(result["min_samples_leaf"]) + " " +
        str(result["accuracy"]) + "\n")
max_result = max(sorted_results, key=lambda x: x["accuracy"])
f.write(str(max_result["n_estimators"]) + " " +
        str(max_result["max_features"]) + " " +
        str(max_result["max_depth"]) + " " +
        str(max_result["min_samples_split"]) + " " +
        str(max_result["min_samples_leaf"]) + " " +
        str(max_result["accuracy"]) + "\n")
print("Optimal: " +
        str(max_result["n_estimators"]) + " " +
        str(max_result["max_features"]) + " " +
        str(max_result["max_depth"]) + " " +
        str(max_result["min_samples_split"]) + " " +
        str(max_result["min_samples_leaf"]) + " " +
        str(max_result["accuracy"]))
f.close()