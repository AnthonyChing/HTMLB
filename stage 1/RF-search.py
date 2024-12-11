from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd

# Load data
df = pd.read_csv('undropped_train.csv')

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

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1126)

f = open('RF-search.txt', 'w')
max = 0
for i in range(100, 2100, 100):
    for j in range(1, 21):
        # Initialize the model
        rf = RandomForestClassifier(n_estimators=i, random_state=j)

        # Train the model
        rf.fit(X_train, y_train)

        # Predict and evaluate
        y_pred = rf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        if accuracy > max:
            max = accuracy
            nBest = i
            rBest = j
        print(f"{i}, {j}\t:", accuracy)
        f.write(str(accuracy) + "\t")
    f.write("\n")
f.write(str(nBest)+"\t"+str(rBest)+"\t"+str(max)+"\n")
f.close()
print(nBest, rBest, max)