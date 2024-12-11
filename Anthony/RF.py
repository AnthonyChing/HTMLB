from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import os

# Load data
df = pd.read_csv('../uranus/preprocessing/undropped_train.csv')

# Fill boolean columns with mode (most frequent value)
bool_columns = ['is_night_game']
for col in bool_columns:
    df[col] = df[col].fillna(df[col].mode()[0]).astype(int)  # Convert boolean to integer

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

n_estimators = 190
random_state = 9
# Initialize the model
rf = RandomForestClassifier(n_estimators = n_estimators, random_state = random_state)

# Train the model
rf.fit(X, y)

# Predict for Stage 1
df = pd.read_csv(r'../stage 1/same_season_test_data.csv')
for col in bool_columns:
    df[col] = df[col].fillna(df[col].mode()[0]).astype(int)
numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns
df[numerical_columns] = df[numerical_columns].fillna(df[numerical_columns].median())

categorical_columns = df.select_dtypes(include=['object']).columns
df[categorical_columns] = df[categorical_columns].fillna("Unknown")
# Apply the same encoding to test data
for col in categorical_columns:
    df[col] = df[col].map({value: idx for idx, value in enumerate(encodings[col])}).fillna(-1).astype(int)
X_test = df.drop(columns=['id']).values
# Predict and evaluate
y_pred = rf.predict(X_test)

if not os.path.exists(r'../stage 1/submissions'):
    os.makedirs(r'../stage 1/submissions')
f = open(rf'../stage 1/submissions/RF-{n_estimators}-{random_state}.csv', 'w')
f.write("id,home_team_win\n")
for index, y_pred in enumerate(y_pred):
    if(y_pred == 1):
        f.write(str(index) + ",True\n")
    else:
        f.write(str(index) + ",False\n")
f.close()

# Predict for Stage 2
df = pd.read_csv(r'../stage 2/2024_test_data.csv')
for col in bool_columns:
    df[col] = df[col].fillna(df[col].mode()[0]).astype(int)
numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns
df[numerical_columns] = df[numerical_columns].fillna(df[numerical_columns].median())

categorical_columns = df.select_dtypes(include=['object']).columns
df[categorical_columns] = df[categorical_columns].fillna("Unknown")
# Apply the same encoding to test data
for col in categorical_columns:
    df[col] = df[col].map({value: idx for idx, value in enumerate(encodings[col])}).fillna(-1).astype(int)
X_test = df.drop(columns=['id']).values
# Predict and evaluate
y_pred = rf.predict(X_test)

if not os.path.exists(r'../stage 2/submissions'):
    os.makedirs(r'../stage 2/submissions')
f = open(rf'../stage 2/submissions/RF-{n_estimators}-{random_state}.csv', 'w')
f.write("id,home_team_win\n")
for index, y_pred in enumerate(y_pred):
    if(y_pred == 1):
        f.write(str(index) + ",True\n")
    else:
        f.write(str(index) + ",False\n")
f.close()