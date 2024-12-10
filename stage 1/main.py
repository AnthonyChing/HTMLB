# %% Read data
# Train model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dropout
from tensorflow.keras.models import load_model
import numpy as np
import pandas as pd
import os

df = pd.read_csv('train.csv')
Data = df.drop(columns=['year', 'month', 'day']).values

N = 10 # The number of lookback rows
X_train = []
y_train = []
for i in range(N,len(Data)):
    X_train.append(Data[i-N:i+1, :])
    y_train.append(Data[i, :])

X_train = np.array(X_train)
y_train = np.array(y_train)
for i in range(len(X_train)):
    X_train[i][-1][0] = -1

# %% Train model
regressor = Sequential ()
regressor.add(LSTM(units = 128, return_sequences = True, input_shape = (X_train.shape[1], X_train.shape[2])))
regressor.add(LSTM(units = 64))

regressor.add(Dropout(0.2))

# output layer
regressor.add(Dense(units = 160))
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

#開始訓練
regressor.fit(X_train, y_train, epochs = 100, batch_size = 128)

#保存模型
from datetime import datetime
NowDateTime = datetime.now().strftime("%Y-%m-%dT%H_%M_%SZ")
regressor.save('LSTM_'+NowDateTime+'.h5')
print('Model Saved')
# %% Predict
regressor = load_model('LSTM_2024-12-06T16_41_39Z.h5')
df = pd.read_csv('test.csv')
df.insert(df.columns.get_loc("home_team_rest"), "home_team_win", -1)
history = pd.read_csv('train.csv')
# print(df)
f = open("re.csv", "w")
lines = ["id,home_team_win\n"]
for i in range(2):
    season = df.at[i, 'season']
    home_team_index = df.at[i, 'home_team_index']
    filtered_df = history[(history['season'] == season) & (history['home_team_index'] == home_team_index)]
    sorted_df = filtered_df.sort_values(by=['year', 'month', 'day'], ascending=[False, False, False])
    sorted_df = sorted_df.drop(columns=['year', 'month', 'day'])
    result = sorted_df.head(N)

    row_to_add = df.iloc[[i]]
    result = pd.concat([result, row_to_add], ignore_index=True)

    X_test = []
    X_test.append(result)
    
    #Reshaping
    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], X_test.shape[2]))
    
    predicted = regressor.predict(X_test)
    print(predicted)
    print(predicted[0][0])
    lines.append(f"{i},{predicted[0][0]}\n")
f.writelines(lines)
f.close()
# %%
