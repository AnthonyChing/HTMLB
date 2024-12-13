import pandas as pd
df = pd.read_csv('../../uranus/preprocessing/undropped_train.csv')
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values(by=['date'], ascending=[True])

splits = 5
N = len(df)
n = int(N/splits)

train_size = [i*n for i in range(1, splits)]

for index, size in enumerate(train_size):
    train = df.head(size)
    if index == splits - 2:
        test = df.iloc[size:N]
    else:
        test = df.iloc[size:size+n]
    train.to_csv(f'../train_data_tscv_{index}.csv', index=False)
    test.to_csv(f'../val_data_tscv_{index}.csv', index=False)