from joblib import load
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
df = pd.read_csv('../uranus/preprocessing/undropped_train.csv')
df = df.drop(columns=['home_team_win', 'date', 'id'])
rf = load("rf_model_tscv_900_log2_10.joblib")
importance = {}
name = df.columns.tolist()
for i in range(len(name)):
    importance[f'{name[i]}'] = rf.feature_importances_[i]
importance = dict(sorted(importance.items(), key=lambda item: item[1]))
with open('importance.txt', 'w') as f:
    for key, values in importance.items():
        print(key, values)
        f.write(str(key) + " " + str(values) + "\n")
