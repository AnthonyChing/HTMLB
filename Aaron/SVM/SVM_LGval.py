#%%
import numpy as np
import scipy as sp
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from category_encoders import *
import os

from libsvm.svmutil import *

# Use late game validation for stage 1 & stage 2
# For stage 1
X_train_1 = pd.read_csv(os.getcwd() +  "/../data/Late_Game_Dataset1/train_data_1_f.csv")
X_val_1 = pd.read_csv(os.getcwd() +  "/../data/Late_Game_Dataset1/val_data_1_f.csv")
y_train_1 = X_train_1['home_team_win']
y_val_1 = X_val_1['home_team_win']
X_train_1 = X_train_1.drop('home_team_win', axis=1)
X_val_1 = X_val_1.drop('home_team_win', axis=1)

# For stage 2
X_train_2 = pd.read_csv(os.getcwd() +  "/../data/Late_Game_Dataset2/train_data_2_f.csv")
X_val_2 = pd.read_csv(os.getcwd() +  "/../data/Late_Game_Dataset2/val_data_2_f.csv")
y_train_2 = X_train_2['home_team_win']
y_val_2 = X_val_2['home_team_win']
X_train_2 = X_train_2.drop('home_team_win', axis=1)
X_val_2 = X_val_2.drop('home_team_win', axis=1)

# Test data
X_test_1 = pd.read_csv(os.getcwd() +  "/../data/Test/stage_1_filled.csv")
X_test_2 = pd.read_csv(os.getcwd() +  "/../data/Test/stage_2_filled.csv")

X_train = [X_train_1, X_train_2]
X_val = [X_val_1, X_val_2]
y_train = [y_train_1, y_train_2]
y_val = [y_val_1, y_val_2]
X_test = [X_test_1, X_test_2]

X_whole = pd.concat((X_train_1, X_val_1))
y_whole = pd.concat((y_train_1, y_val_1))


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
    X_train[stage] = X_train[stage].drop(drop, axis=1)
    X_val[stage] = X_val[stage].drop(drop, axis=1)
    X_whole = X_whole.drop(drop, axis=1)

X_test[stage] = X_test[stage].drop('id', axis=1)

columns_not_in_B = [col for col in X_test[stage].columns if col not in X_train[stage].columns]

print("Columns in A but not in B:", columns_not_in_B)

#%%
# Encode True/False
X_train[stage]['is_night_game'] = X_train[stage]['is_night_game'].replace({True: 1, False: 0})
X_val[stage]['is_night_game'] = X_val[stage]['is_night_game'].replace({True: 1, False: 0})
X_test[stage]['is_night_game'] = X_test[stage]['is_night_game'].replace({True: 1, False: 0})
X_whole['is_night_game'] = X_whole['is_night_game'].replace({True: 1, False: 0})

y_train[stage] = y_train[stage].replace({True: 1, False: 0})
y_val[stage] = y_val[stage].replace({True: 1, False: 0})
y_whole = y_whole.replace({True: 1, False: 0})

columns_to_encode = ['home_team_abbr', 'away_team_abbr', 'home_pitcher', 'away_pitcher']

encoder = TargetEncoder(cols=columns_to_encode, return_df=True)
target = y_train[stage]

X_train[stage] = encoder.fit_transform(X_train[stage], target)
X_val[stage] = encoder.transform(X_val[stage])

X_whole = encoder.fit_transform(X_whole, y_whole)
X_test[stage] = encoder.transform(X_test[stage])

print(X_train[stage].shape)
print(X_val[stage].shape)
print(X_test[stage].shape)
print(X_whole.shape)


#%%
# Scale data
scaler = MinMaxScaler()

X = scaler.fit_transform(X_train[stage].to_numpy())
y = y_train[stage].to_numpy().flatten()

X_v = scaler.fit_transform(X_val[stage].to_numpy())
y_v = y_val[stage].to_numpy().flatten()

X_t = scaler.fit_transform(X_test[stage].to_numpy())

X_w = scaler.fit_transform(X_whole.to_numpy())
y_w = y_whole.to_numpy().flatten()


# print(type(X_train[stage]), type(y_train[stage]))
# print(X_train[stage].shape, y_train[stage].shape)
# print(type(X_val[stage]), type(y_val[stage]))
# print(X_val[stage].shape, y_val[stage].shape)
# print(type(X_test[stage]))
# print(X_test[stage].shape)

#%%time
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
        m = svm_train(y, X, f"-s 0 -t 2 -g {gamma} -c {C} -h 0 -m 1000")
        p_label, p_acc, p_val = svm_predict(y_v, X_v, m)
        acc[i][j] = p_acc[0]
        if(p_acc[0] > acc_opt):
            acc_opt = p_acc[0]
            gamma_opt = gamma
            C_opt = C
print(gamma_opt, C_opt, acc_opt, f"time: {time.time() - start_time}")

f = open(f'gaussian_acc_spread_S{stage + 1}_N{N}_{col}col_base2.txt', 'w')
for i in range(0, N + 1):
    for j in range(0, N + 1):
        f.write(f"{acc[i][j]} ")
    f.write("\n")
f.flush()

#%%time

m = svm_train(y_w, X_w, f"-s 0 -t 2 -g {gamma_opt} -c {C_opt} -h 0 -m 1000")
svm_save_model(f'models/RBF_C{C_opt}_G{gamma_opt}_LateGame_S{stage + 1}_{col}col.model', m)
p_label, p_acc, p_val = svm_predict(y_v, X_v, m)

# %%
# Predict
m = svm_load_model(f'models/RBF_C{C_opt}_G{gamma_opt}_LateGame_S{stage + 1}_{col}col.model')
p_labs, p_acc, p_vals = svm_predict([], X_t, m)

tmp = [bool(x) for x in p_labs]

predict = pd.DataFrame()
predict['home_team_win'] = tmp
predict.to_csv(f"predictions/RBF_C{C_opt}_G{gamma_opt}_LateGame_S{stage + 1}_{col}col.csv", index_label="id")



# %%
