from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import pandas as pd
import os

# Load data
df = pd.read_csv('../uranus/preprocessing/undropped_train.csv')

# Fill boolean columns with mode (most frequent value)
df['is_night_game'] = df['is_night_game'].apply(lambda x: 1 if x == True else 0 if x == False else x)

# Fill categorical columns with "Unknown"
categorical_columns = df.select_dtypes(include=['object']).columns
df[categorical_columns] = df[categorical_columns].fillna("Unknown")

# Encode categorical columns (factorize to numeric codes)
encodings = {}
for col in categorical_columns:
    df[col], encodings[col] = pd.factorize(df[col])

# Define features and target
X = df.drop(columns=['home_team_win', 'date', 'id']).values  # Features
y = df['home_team_win'].values  # Target

# Define parameter grid
param_grid = {
    'n_estimators': [200, 300, 400],
    'max_features': ['sqrt', 'log2'],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
# Initialize GridSearchCV
grid_search = GridSearchCV(
    estimator=RandomForestClassifier(oob_score=True, random_state=42),
    param_grid=param_grid,
    scoring='accuracy',
    cv=5,  # Cross-validation
    n_jobs=-1,  # Use all available CPU cores
    verbose=2
)
# Perform grid search
grid_search.fit(X, y)

# Best parameters and OOB score
print("Best Parameters:", grid_search.best_params_)
print("Best OOB Score:", grid_search.best_estimator_.oob_score_)

# Get the best model from grid search
best_rf = grid_search.best_estimator_

# Predict for Stage 1
df = pd.read_csv(r'../stage 1/same_season_test_data.csv')
df['is_night_game'] = df['is_night_game'].apply(lambda x: 1 if x == True else 0 if x == False else x)

categorical_columns = df.select_dtypes(include=['object']).columns
df[categorical_columns] = df[categorical_columns].fillna("Unknown")
# Apply the same encoding to test data
for col in categorical_columns:
    df[col] = df[col].map({value: idx for idx, value in enumerate(encodings[col])}).fillna(-1).astype(int)
X_test = df.drop(columns=['id']).values
# Predict and evaluate
y_pred = best_rf.predict(X_test)

if not os.path.exists(r'../stage 1/submissions'):
    os.makedirs(r'../stage 1/submissions')
f = open(rf'../stage 1/submissions/RF-grid.csv', 'w')
f.write("id,home_team_win\n")
for index, y_pred in enumerate(y_pred):
    if(y_pred == 1):
        f.write(str(index) + ",True\n")
    else:
        f.write(str(index) + ",False\n")
f.close()

# Predict for Stage 2
df = pd.read_csv(r'../stage 2/2024_test_data.csv')
df['is_night_game'] = df['is_night_game'].apply(lambda x: 1 if x == True else 0 if x == False else x)

categorical_columns = df.select_dtypes(include=['object']).columns
df[categorical_columns] = df[categorical_columns].fillna("Unknown")
# Apply the same encoding to test data
for col in categorical_columns:
    df[col] = df[col].map({value: idx for idx, value in enumerate(encodings[col])}).fillna(-1).astype(int)
X_test = df.drop(columns=['id']).values
# Predict and evaluate
y_pred = best_rf.predict(X_test)

if not os.path.exists(r'../stage 2/submissions'):
    os.makedirs(r'../stage 2/submissions')
f = open(rf'../stage 2/submissions/RF-grid.csv', 'w')
f.write("id,home_team_win\n")
for index, y_pred in enumerate(y_pred):
    if(y_pred == 1):
        f.write(str(index) + ",True\n")
    else:
        f.write(str(index) + ",False\n")
f.close()