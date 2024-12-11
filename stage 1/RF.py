from sklearn.ensemble import RandomForestClassifier
import pandas as pd

# Load data
df = pd.read_csv('undropped_train.csv')

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

n_estimators = 400
random_state = 14
# Initialize the model
rf = RandomForestClassifier(n_estimators = n_estimators, random_state = random_state)

# Train the model
rf.fit(X, y)

df = pd.read_csv('raw/same_season_test_data.csv')
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

f = open('RF.csv', 'w')
f.write("id,home_team_win\n")
for index, y_pred in enumerate(y_pred):
    if(y_pred == 1):
        f.write(str(index) + ",True\n")
    else:
        f.write(str(index) + ",False\n")
f.close()