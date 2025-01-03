from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import pandas as pd
import os
from sklearn.metrics import accuracy_score

# Load data
df = pd.read_csv('../uranus/preprocessing/undropped_train.csv')

categorical_columns = df.select_dtypes(include=['object']).columns
categorical_columns = categorical_columns.drop('is_night_game')

# Encode categorical columns (factorize to numeric codes)
encodings = {}
for col in categorical_columns:
    df[col], encodings[col] = pd.factorize(df[col])

# Define features and target
X = df.drop(columns=['home_team_win', 'date', 'id']).values  # Features
y = df['home_team_win'].values  # Target

# rf_late_1 = RandomForestClassifier(n_estimators=700, 
#                             max_features='sqrt',
#                             max_depth=15,
#                             random_state=1126)
# rf_late_1.fit(X, y)
# print('trained rf_late_1')

# rf_late_2 = RandomForestClassifier(n_estimators=300, 
#                             max_features='log2',
#                             max_depth=10,
#                             random_state=1126)
# rf_late_2.fit(X, y)
# print('trained rf_late_2')

rf_tscv = RandomForestClassifier(n_estimators=900,
                            max_features='log2',
                            max_depth=10,
                            random_state=1126)
rf_tscv.fit(X, y)
print('trained rf_tscv')

# Define parameter grid
# param_grid = {
#     'n_estimators': [350, 400, 450],
#     'max_features': ['sqrt', 'log2'],
#     'max_depth': [5, 10, 15],
#     'min_samples_split': [2, 5, 10],
#     'min_samples_leaf': [1, 2, 4]
# }
# Initialize GridSearchCV
# grid_search = GridSearchCV(
#     estimator=RandomForestClassifier(oob_score=True, random_state=42),
#     param_grid=param_grid,
#     scoring='accuracy',
#     cv=5,  # Cross-validation
#     n_jobs=-1,  # Use all available CPU cores
#     verbose=2
# )
# Perform grid search
# grid_search.fit(X, y)

# Best parameters and OOB score
# print("Best Parameters:", grid_search.best_params_)
# print("Best OOB Score:", grid_search.best_estimator_.oob_score_)
# Best Parameters: {'max_depth': 10, 'max_features': 'log2', 'min_samples_leaf': 1, 'min_samples_split': 5, 'n_estimators': 400}
# Best OOB Score: 0.5513689346706424
# Best Parameters: {'max_depth': 10, 'max_features': 'log2', 'min_samples_leaf': 4, 'min_samples_split': 10, 'n_estimators': 10000}
# Best OOB Score: 0.5548025661877655
# Best Parameters: {'max_depth': 5, 'max_features': 'log2', 'min_samples_leaf': 2, 'min_samples_split': 2, 'n_estimators': 350}
# Best OOB Score: 0.5595012198427758
# Get the best model from grid search
# best_rf = grid_search.best_estimator_

# Predict for Stage 1
df = pd.read_csv(r'../stage 1/same_season_test_data.csv')
ans = pd.read_csv(r'../stage 1/same_season_test_label.csv')
categorical_columns = df.select_dtypes(include=['object']).columns
categorical_columns = categorical_columns.drop('is_night_game')
# Apply the same encoding to test data
for col in categorical_columns:
    df[col] = df[col].map({value: idx for idx, value in enumerate(encodings[col])})
X_test = df.drop(columns=['id']).values
# Predict and evaluate
y_pred_tscv = rf_tscv.predict(X_test)
y_ans = ans["home_team_win"].replace({True: 1, False: 0}).values
acc_tscv = accuracy_score(y_ans, y_pred_tscv)
print(1 - acc_tscv)

if not os.path.exists(r'../stage 1/submissions'):
    os.makedirs(r'../stage 1/submissions')

# with open(f'../stage 1/submissions/RF-late.csv', 'w') as f:
#     f.write("id,home_team_win\n")
#     for index, y_pred in enumerate(y_pred_late):
#         if(y_pred == 1):
#             f.write(str(index) + ",True\n")
#         else:
#             f.write(str(index) + ",False\n")

with open(f'../stage 1/submissions/RF.csv', 'w') as f:
    f.write("id,home_team_win\n")
    for index, y_pred in enumerate(y_pred_tscv):
        if(y_pred == 1):
            f.write(str(index) + ",True\n")
        else:
            f.write(str(index) + ",False\n")

# Predict for Stage 2
df = pd.read_csv(r'../stage 2/2024_test_data.csv')
ans = pd.read_csv(r'../stage 2/2024_test_label.csv')
# Apply the same encoding to test data
for col in categorical_columns:
    df[col] = df[col].map({value: idx for idx, value in enumerate(encodings[col])})
X_test = df.drop(columns=['id']).values
# Predict and evaluate
y_pred_tscv = rf_tscv.predict(X_test)
y_ans = ans["home_team_win"].replace({True: 1, False: 0}).values
acc_tscv = accuracy_score(y_ans, y_pred_tscv)
print(1 - acc_tscv)

# if not os.path.exists(r'../stage 2/submissions'):
#     os.makedirs(r'../stage 2/submissions')

# with open(f'../stage 2/submissions/RF-late.csv', 'w') as f:
#     f.write("id,home_team_win\n")
#     for index, y_pred in enumerate(y_pred_late):
#         if(y_pred == 1):
#             f.write(str(index) + ",True\n")
#         else:
#             f.write(str(index) + ",False\n")

with open(f'../stage 2/submissions/RF.csv', 'w') as f:
    f.write("id,home_team_win\n")
    for index, y_pred in enumerate(y_pred_tscv):
        if(y_pred == 1):
            f.write(str(index) + ",True\n")
        else:
            f.write(str(index) + ",False\n")