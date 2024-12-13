from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
from joblib import Parallel, delayed

# Load data
df = pd.read_csv('../uranus/preprocessing/undropped_train.csv')

# Fill boolean columns with mode (most frequent value)
bool_columns = ['is_night_game', 'home_team_win']
for col in bool_columns:
    df[col] = df[col].fillna(df[col].mode()[0]).astype(int)  # Convert boolean to integer

# Fill categorical columns with "Unknown"
categorical_columns = df.select_dtypes(include=['object']).columns
df[categorical_columns] = df[categorical_columns].fillna("Unknown")

# Encode categorical columns (factorize to numeric codes)
for col in categorical_columns:
    df[col] = pd.factorize(df[col])[0]

# Define features and target
X = df.drop(columns=['home_team_win', 'date', 'id']).values  # Features
y = df['home_team_win'].values  # Target

start = 100
end = 400
step = 15
# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1126)
params_list = [{"i": i, "j": j} for i in range(start, end+step, step) for j in range(1, 21)]

def calc(params):
    rf = RandomForestClassifier(n_estimators=params["i"], random_state=params["j"])

    # Train the model
    rf.fit(X_train, y_train)

    # Predict and evaluate
    y_pred = rf.predict(X_test)
    print(params["i"], params["j"], accuracy_score(y_test, y_pred))
    return (params["i"], params["j"], accuracy_score(y_test, y_pred))


results = Parallel(n_jobs=36)(delayed(calc)(params) for params in params_list)
f = open(f'RF-search-{start}-{end}.txt', 'w')
sorted_results = sorted(results, key=lambda x: (x[0], x[1]))
for accuracy in sorted_results:
    f.write(str(accuracy) + "\n")
max_result = max(sorted_results, key=lambda x: x[2])
f.write(str(max_result[0]) + " " + str(max_result[1]) + " " + str(max_result[2]) + "\n")
print(max_result[0], max_result[1], max_result[2])
f.close()