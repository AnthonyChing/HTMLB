import pandas as pd
import numpy as np
import random
from sklearn.preprocessing import StandardScaler
import json

# gradient E_in(w) = 1/N sum[n=1..N] sigmoid(-yw^Tx_n)(-y_nx_n)

N =  100000 # times to train the model
lr = 0.002 # fixed learning rate 
def sign(a):
    return 1 if a > 0 else -1
def accuracy(w, x, y):
    correct = 0
    for i in range(len(x)):
        if sign(y[i]) == sign(np.dot(w, x[i])):
            correct += 1
    print(f"Accuracy: {float(100 * correct / len(x)):.2f}%")
    
def logistic_regression(w, x, y, N, lr): # logistic regression by SGD
    def sigmoid(w, x, y) -> float:
        z = -y * np.dot(w, x)
        return 1 / (1 + np.exp(z))
    
    # stochastic gradient descent
    for times in range(N):
        rand = random.randint(0, len(x) - 1)
        error = sigmoid(w, x[rand], -y[rand])
        for i in range(len(w)):
            w[i] += (lr * error * (y[rand] * x[rand][i]))
    return w
    
train_data_path = 'final_train.csv'
data = pd.read_csv(train_data_path)

float_cols = data.select_dtypes(include = ['float64'])
new_data = pd.DataFrame(float_cols)
new_data = new_data.apply(lambda col: col.fillna(col.median()))
#new_data = new_data.drop(columns = ['home_batting_batting_avg_mean', 'home_batting_onbase_perc_mean', 
                                    # 'home_batting_onbase_plus_slugging_mean', 'home_batting_leverage_index_avg_mean',
                                    # 'home_batting_wpa_bat_mean', 'home_batting_RBI_mean',
                                    # 'home_batting_batting_avg_std', 'home_batting_onbase_perc_std', 
                                    # 'home_batting_onbase_plus_slugging_std', 'home_batting_leverage_index_avg_std',
                                    # 'home_batting_wpa_bat_std', 'home_batting_RBI_std',
                                    # 'home_batting_batting_avg_skew', 'home_batting_onbase_perc_skew', 
                                    # 'home_batting_onbase_plus_slugging_skew', 'home_batting_leverage_index_avg_skew',
                                    # 'home_batting_wpa_bat_skew', 'home_batting_RBI_skew',
                                    # 'away_batting_batting_avg_mean', 'away_batting_onbase_perc_mean', 
                                    # 'away_batting_onbase_plus_slugging_mean', 'away_batting_leverage_index_avg_mean',
                                    # 'away_batting_wpa_bat_mean', 'away_batting_RBI_mean',
                                    # 'away_batting_batting_avg_std', 'away_batting_onbase_perc_std', 
                                    # 'away_batting_onbase_plus_slugging_std', 'away_batting_leverage_index_avg_std',
                                    # 'away_batting_wpa_bat_std', 'away_batting_RBI_std',
                                    # 'away_batting_batting_avg_skew', 'away_batting_onbase_perc_skew', 
                                    # 'away_batting_onbase_plus_slugging_skew', 'away_batting_leverage_index_avg_skew',
                                    # 'away_batting_wpa_bat_skew', 'away_batting_RBI_skew',
                                    # 'home_pitching_earned_run_avg_mean', 'home_pitching_SO_batters_faced_mean',
                                    # 'home_pitching_H_batters_faced_mean', 'home_pitching_BB_batters_faced_mean',
                                    # 'home_pitching_leverage_index_avg_mean', 'home_pitching_wpa_def_mean',
                                    # 'home_pitching_earned_run_avg_std', 'home_pitching_SO_batters_faced_std',
                                    # 'home_pitching_H_batters_faced_std', 'home_pitching_BB_batters_faced_std',
                                    # 'home_pitching_leverage_index_avg_std', 'home_pitching_wpa_def_std',
                                    # 'home_pitching_earned_run_avg_skew', 'home_pitching_SO_batters_faced_skew',
                                    # 'home_pitching_H_batters_faced_skew', 'home_pitching_BB_batters_faced_skew',
                                    # 'home_pitching_leverage_index_avg_skew', 'home_pitching_wpa_def_skew',
                                    # 'away_pitching_earned_run_avg_mean', 'away_pitching_SO_batters_faced_mean',
                                    # 'away_pitching_H_batters_faced_mean', 'away_pitching_BB_batters_faced_mean',
                                    # 'away_pitching_leverage_index_avg_mean', 'away_pitching_wpa_def_mean',
                                    # 'away_pitching_earned_run_avg_std', 'away_pitching_SO_batters_faced_std',
                                    # 'away_pitching_H_batters_faced_std', 'away_pitching_BB_batters_faced_std',
                                    # 'away_pitching_leverage_index_avg_std', 'away_pitching_wpa_def_std',
                                    # 'away_pitching_earned_run_avg_skew', 'away_pitching_SO_batters_faced_skew',
                                    # 'away_pitching_H_batters_faced_skew', 'away_pitching_BB_batters_faced_skew',
                                    # 'away_pitching_leverage_index_avg_skew', 'away_pitching_wpa_def_skew'])

y = np.array(data['home_team_win'], dtype=int)
y = [2 * yi - 1 for yi in y]

y_start = 0
y_end = 0

unique_years = new_data['season'].unique()
result = []

# x = new_data.values.tolist()
# x = np.delete(x, 4, axis = 1)
# w = np.zeros(len(x[0]))
# w = logistic_regression(w, x, y, N, lr)
# accuracy(w, x, y)

for year in unique_years:
    print(year)
    x = new_data[new_data['season'] == year].values.tolist()
    x = np.delete(x, 4, axis = 1)
    scaler = StandardScaler()
    x = scaler.fit_transform(x)
    y_end = y_start + len(x)
    w = np.zeros(len(x[0]))
    w = logistic_regression(w, x, y[y_start : y_end], N, lr)
    accuracy(w, x, y[y_start : y_end])
    result.append(w.tolist())
    y_start += len(x)

# with open("same_season_result.json", "w") as f:
#     json.dump(result, f)
    