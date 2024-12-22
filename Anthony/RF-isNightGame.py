from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import os
from sklearn.metrics import accuracy_score

# Load data
df = pd.read_csv('../uranus/preprocessing/undropped_train.csv')
df = df.dropna(subset=['is_night_game'])

categorical_columns = df.select_dtypes(include=['object']).columns
categorical_columns = categorical_columns.drop('is_night_game')
# Encode categorical columns (factorize to numeric codes)
encodings = {}
for col in categorical_columns:
    df[col], encodings[col] = pd.factorize(df[col])

# Define features and target
X = df.drop(columns=['is_night_game', 'date', 'id']).values  # Features
y = df['is_night_game'].replace({'True': True, 'False': False}).values  # Target
print(df['is_night_game'].unique())
print(df['is_night_game'].dtype)

rf_tscv = RandomForestClassifier(n_estimators=900,
                            max_features='log2',
                            max_depth=10,
                            random_state=1126)
rf_tscv.fit(X, y)
print('trained tscv')

# Predict for Stage 1
df = pd.read_csv(r'../stage 1/same_season_test_data.csv')
df = df.dropna(subset=['is_night_game'])
ans = pd.read_csv('../Aaron/is_night_game/isnightgame_stage_1_label.csv')
categorical_columns = df.select_dtypes(include=['object']).columns
categorical_columns = categorical_columns.drop('is_night_game')
# Apply the same encoding to test data
for col in categorical_columns:
    df[col] = df[col].map({value: idx for idx, value in enumerate(encodings[col])})
X_test = df.drop(columns=['id']).values
# Predict and evaluate
y_pred_tscv = rf_tscv.predict(X_test)
y_ans = ans["is_night_game"].replace({True: 1, False: 0}).values
acc_tscv = accuracy_score(y_ans, y_pred_tscv)
print(1 - acc_tscv)

if not os.path.exists(r'../stage 1/submissions'):
    os.makedirs(r'../stage 1/submissions')

with open(f'../stage 1/submissions/RF-is_night_game.csv', 'w') as f:
    f.write("id,is_night_game\n")
    for index, y_pred in enumerate(y_pred_tscv):
        if(y_pred == 1):
            f.write(str(index) + ",True\n")
        else:
            f.write(str(index) + ",False\n")

# Predict for Stage 2
df = pd.read_csv(r'../stage 2/2024_test_data.csv')
df = df.dropna(subset=['is_night_game'])
ans = pd.read_csv('../Aaron/is_night_game/isnightgame_stage_2_label.csv')
# Apply the same encoding to test data
for col in categorical_columns:
    df[col] = df[col].map({value: idx for idx, value in enumerate(encodings[col])})
X_test = df.drop(columns=['id']).values
# Predict and evaluate
y_pred_tscv = rf_tscv.predict(X_test)
y_ans = ans["is_night_game"].replace({True: 1, False: 0}).values
acc_tscv = accuracy_score(y_ans, y_pred_tscv)
print(1 - acc_tscv)

if not os.path.exists(r'../stage 2/submissions'):
    os.makedirs(r'../stage 2/submissions')

with open(f'../stage 2/submissions/RF-is_night_game.csv', 'w') as f:
    f.write("id,is_night_game\n")
    for index, y_pred in enumerate(y_pred_tscv):
        if(y_pred == 1):
            f.write(str(index) + ",True\n")
        else:
            f.write(str(index) + ",False\n")