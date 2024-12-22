from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
from joblib import Parallel, delayed, dump

# Load data
df = pd.read_csv('../uranus/preprocessing/undropped_train.csv')

categorical_columns = df.select_dtypes(include=['object']).columns
categorical_columns = categorical_columns.drop('is_night_game')

# Encode categorical columns (factorize to numeric codes)
encodings = {}
for col in categorical_columns:
    df[col], encodings[col] = pd.factorize(df[col])

X_train_tscv = []
X_val_tscv = []
y_train_tscv = []
y_val_tscv = []

for i in range(4):
    train_data = pd.read_csv(f'train_data_tscv_{i}.csv')
    val_data = pd.read_csv(f'val_data_tscv_{i}.csv')

    for col in categorical_columns:
        train_data[col] = train_data[col].map({value: idx for idx, value in enumerate(encodings[col])})
        val_data[col] = val_data[col].map({value: idx for idx, value in enumerate(encodings[col])})
    
    X_train = train_data.drop(columns=['home_team_win', 'date', 'id']).values  # Features
    y_train = train_data['home_team_win'].values  # Target

    X_val = val_data.drop(columns=['home_team_win', 'date', 'id']).values  # Features
    y_val = val_data['home_team_win'].values  # Target

    X_train_tscv.append(X_train)
    X_val_tscv.append(X_val)
    y_train_tscv.append(y_train)
    y_val_tscv.append(y_val)

n_estimators = [100 * i for i in range(1, 3, 2)]
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
    acc = []
    for i in range(4):
        rf = RandomForestClassifier(n_estimators=params["n_estimators"], 
                                max_features=params["max_features"],
                                max_depth=params["max_depth"],
                                oob_score=True,
                                random_state=1126)
        rf.fit(X_train_tscv[i], y_train_tscv[i])
        y_pred = rf.predict(X_val_tscv[i])
        acc.append(accuracy_score(y_val_tscv[i], y_pred))
        print(f"Calculated acc[{i}]: " + toString(params) + str(acc[i]))

    accuracy = (acc[0]*1+acc[1]*2+acc[2]*3+acc[3]*4)/10
    params["accuracy"] = accuracy
    params["oob_score"] = rf.oob_score_
    params["feature_importance"] = rf.feature_importances_
    params["model"] = rf
    print(toString(params) + str(accuracy))
    
    return params

results = Parallel(n_jobs=8)(delayed(calc)(params) for params in params_list)
max_features_priority = {"sqrt": 1, "log2": 2, None: 3}

with open(f'RF-search-tscv-grid-less.txt', 'w') as f:
    sorted_results = sorted(
        results,
        key=lambda x: (
            x["n_estimators"],
            max_features_priority.get(x["max_features"], float('inf')),
            x["max_depth"] if x["max_depth"] is not None else 0
        )
    )
    for result in sorted_results:
        f.write(toString(result) + " " + str(result["accuracy"]) + " " + str(result["oob_score"]) + "\n")
        model = result["model"]
        filename = f"rf_model_tscv_{result['n_estimators']}_{result['max_features']}_{result['max_depth']}.joblib"
        dump(model, filename)
        print(f"Model saved to {filename}")
    max_result = max(sorted_results, key=lambda x: x["accuracy"])
    f.write(toString(max_result) + " " + str(max_result["accuracy"]) + " " + str(max_result["oob_score"]) + "\n")
    print("Optimal:", toString(max_result), max_result["accuracy"], max_result["oob_score"])
    print(max_result["feature_importance"])