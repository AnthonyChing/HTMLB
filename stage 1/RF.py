from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
df = pd.read_csv('undropped_train.csv')

# Load your data
X = df.drop(columns=['home_team_win', 'date', 'id']).values  # Features (e.g., team stats, player stats)
y = df['home_team_win'].values  # Target (e.g., win or lose)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the model
rf = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
rf.fit(X_train, y_train)

# Predict and evaluate
y_pred = rf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
