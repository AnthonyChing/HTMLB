#%%
import numpy as np
import scipy as sp
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from category_encoders import *
import os

from libsvm.svmutil import *

# Use time series cv for stage 1 & stage 2
X_train = [pd.DataFrame() for _ in range(4)]
X_val =[pd.DataFrame() for _ in range(4)]
y_train = [pd.Series() for _ in range(4)]
y_val = [pd.Series() for _ in range(4)]
for i in range(4):
    X_train[i] = pd.read_csv(os.getcwd() + f"/../data/TSCV/train_data_tscv_{i}_f.csv")
    X_val[i] = pd.read_csv(os.getcwd() + f"/../data/TSCV/val_data_tscv_{i}_f.csv")
    y_train[i] = X_train[i]['home_team_win']
    y_val[i] = X_val[i]['home_team_win']
    X_train[i] = X_train[i].drop('home_team_win', axis=1)
    X_val[i] = X_val[i].drop('home_team_win', axis=1)

# Test data
X_test_1 = pd.read_csv(os.getcwd() +  "/../data/Test/stage_1_filled.csv")
X_test_2 = pd.read_csv(os.getcwd() +  "/../data/Test/stage_2_filled.csv")
X_test = [X_test_1, X_test_2]

X_whole = pd.concat((X_train[3], X_val[3]))
y_whole = pd.concat((y_train[3], y_val[3]))

for i in range(4):
    print(X_train[i].shape)
    print(X_val[i].shape)
print(X_test[stage].shape)
print(X_whole.shape)

#%%
# Stage 1 or 2
stage = 1

#%%
drop_std_skew = 0
# Drop all _std and _skew
if(drop_std_skew):
    X_train[stage] = X_train[stage].loc[:, ~X_train[stage].columns.str.endswith(('_std', '_skew'))]
    X_val[stage] = X_val[stage].loc[:, ~X_val[stage].columns.str.endswith(('_std', '_skew'))]
    X_test[stage] = X_test[stage].loc[:, ~X_test[stage].columns.str.endswith(('_std', '_skew'))]
    X_whole[stage] = X_whole[stage].loc[:, ~X_whole[stage].columns.str.endswith(('_std', '_skew'))]
    col = 78
else:
    col = 162

# Drop columns
for drop in ['home_team_season', 'away_team_season', 'date', 'id']:
    for i in range(4):
        X_train[i] = X_train[i].drop(drop, axis=1)
        X_val[i] = X_val[i].drop(drop, axis=1)
    X_whole = X_whole.drop(drop, axis=1)

X_test[stage] = X_test[stage].drop('id', axis=1)

#%%
# Encode True/False
for i in range(4):
    X_train[i]['is_night_game'] = X_train[i]['is_night_game'].replace({True: 1, False: 0})
    X_val[i]['is_night_game'] = X_val[i]['is_night_game'].replace({True: 1, False: 0})

X_test[stage]['is_night_game'] = X_test[stage]['is_night_game'].replace({True: 1, False: 0})
X_whole['is_night_game'] = X_whole['is_night_game'].replace({True: 1, False: 0})

for i in range(4):
    y_train[i] = y_train[i].replace({True: 1, False: 0})
    y_val[i] = y_val[i].replace({True: 1, False: 0})
y_whole = y_whole.replace({True: 1, False: 0})


columns_to_encode = ['home_team_abbr', 'away_team_abbr', 'home_pitcher', 'away_pitcher']
for i in range(4):
    encoder = TargetEncoder(cols=columns_to_encode, return_df=True)
    target = y_train[i]
    X_train[i] = encoder.fit_transform(X_train[i], target)
    X_val[i] = encoder.transform(X_val[i])

X_whole = encoder.fit_transform(X_whole, y_whole)
X_test[stage] = encoder.transform(X_test[stage])

for i in range(4):
    print(X_train[i].shape)
    print(X_val[i].shape)
print(X_test[stage].shape)
print(X_whole.shape)



#%%
# Scale data
scaler = MinMaxScaler()
for i in range(4):
    X_train[i] = scaler.fit_transform(X_train[i].to_numpy())
    y_train[i] = y_train[i].to_numpy().flatten()

    X_val[i] = scaler.fit_transform(X_val[i].to_numpy())
    y_val[i] = y_val[i].to_numpy().flatten()

X_t = scaler.fit_transform(X_test[stage].to_numpy())

X_w = scaler.fit_transform(X_whole.to_numpy())
y_w = y_whole.to_numpy().flatten()

# print(type(X_train[stage]), type(y_train[stage]))
# print(X_train[stage].shape, y_train[stage].shape)
# print(type(X_val[stage]), type(y_val[stage]))
# print(X_val[stage].shape, y_val[stage].shape)
# print(type(X_test[stage]))
# print(X_test[stage].shape)
#%%
import time
start_time = time.time()
# Gaussian kernel
N = 7
acc = [[0 for a in range(N + 1)] for b in range(N + 1)] 
gamma_opt = -1
C_opt = -1
acc_opt = -1
for i, gamma in enumerate(np.logspace(-N, 0, num=N + 1, base=2)):
    for j, C in enumerate(np.logspace(-N, 0, num=N + 1, base=2)):
        acc_total = 0
        for v in range(4):
            m = svm_train(y_train[v], X_train[v], f"-s 0 -t 2 -g {gamma} -c {C} -h 0 -m 1000")
            p_label, p_acc, p_val = svm_predict(y_val[v], X_val[v], m)
            acc_total += p_acc[0] * (v + 1)
        acc_total /= 10
        print(f"G:{gamma}, C:{C}, Acc:{acc_total}\n")
        acc[i][j] = acc_total
        if(acc_total > acc_opt):
            acc_opt = acc_total
            gamma_opt = gamma
            C_opt = C
print(gamma_opt, C_opt, acc_opt, f"time: {time.time() - start_time}")

f = open(f'gaussian_acc_spread_S{stage + 1}_N{N}_{col}col_TSCV.txt', 'w')
for i in range(0, N + 1):
    for j in range(0, N + 1):
        f.write(f"{acc[i][j]} ")
    f.write("\n")
f.flush()

#%%
m = svm_train(y_w, X_w, f"-s 0 -t 2 -g {gamma_opt} -c {C_opt} -h 0 -m 1000")
svm_save_model(f'models/RBF_C{C_opt}_G{gamma_opt}_TSCV_S{stage + 1}_{col}col.model', m)


# %%
# Predict
m = svm_load_model(f'models/RBF_C{C_opt}_G{gamma_opt}_TSCV_S{stage + 1}_{col}col.model')
p_labs, p_acc, p_vals = svm_predict([], X_t, m)

tmp = [bool(x) for x in p_labs]

predict = pd.DataFrame()
predict['home_team_win'] = tmp
predict.to_csv(f"predictions/RBF_C{C_opt}_G{gamma_opt}_TSCV_S{stage + 1}_{col}col.csv", index_label="id")



# %%
