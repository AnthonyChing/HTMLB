from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import pandas as pd
import os
from sklearn.metrics import accuracy_score

## Small

# Load data
df_train = pd.read_csv('../Aaron/data_dropping/33%/train_data_33%_tscv_3.csv')
df_val = pd.read_csv('../Aaron/data_dropping/33%/val_data_33%_tscv_3.csv')
df = pd.concat((df_train, df_val))

categorical_columns = df.select_dtypes(include=['object']).columns
categorical_columns = categorical_columns.drop('is_night_game')
# Encode categorical columns (factorize to numeric codes)
encodings = {}
for col in categorical_columns:
    df[col], encodings[col] = pd.factorize(df[col])

# Define features and target
X = df.drop(columns=['home_team_win', 'date', 'id']).values  # Features
y = df['home_team_win'].values  # Target

rf_tscv = RandomForestClassifier(n_estimators=900,
                            max_features='log2',
                            max_depth=10,
                            random_state=1126)
rf_tscv.fit(X, y)
print('trained small')

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

## Medium

# Load data
df_train = pd.read_csv('../Aaron/data_dropping/66%/train_data_66%_tscv_3.csv')
df_val = pd.read_csv('../Aaron/data_dropping/66%/val_data_66%_tscv_3.csv')
df = pd.concat((df_train, df_val))

categorical_columns = df.select_dtypes(include=['object']).columns
categorical_columns = categorical_columns.drop('is_night_game')
# Encode categorical columns (factorize to numeric codes)
encodings = {}
for col in categorical_columns:
    df[col], encodings[col] = pd.factorize(df[col])

# Define features and target
X = df.drop(columns=['home_team_win', 'date', 'id']).values  # Features
y = df['home_team_win'].values  # Target

rf_tscv = RandomForestClassifier(n_estimators=900,
                            max_features='log2',
                            max_depth=10,
                            random_state=1126)
rf_tscv.fit(X, y)
print('trained medium')

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