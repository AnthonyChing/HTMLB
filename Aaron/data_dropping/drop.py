import pandas as pd
import os

def TSCV(df, name):
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
        train.to_csv(f'{name}/train_data_{name}_tscv_{index}.csv', index=False)
        test.to_csv(f'{name}/val_data_{name}_tscv_{index}.csv', index=False)

# Replace with your path
train_data = pd.read_csv(os.getcwd() + '/../../uranus/preprocessing/undropped_train.csv')
stage_1 = pd.read_csv(os.getcwd() + '/../../stage 1/same_season_test_data.csv')
stage_2 = pd.read_csv(os.getcwd() + '/../../stage 2/2024_test_data.csv')

data = [train_data, stage_1, stage_2]
name = ['train_data', 'stage_1_test', 'stage_2_test']

for name, d in zip(name, data):
    dataset1 = []
    dataset2 = []

    grouped = d.groupby(['home_team_abbr', 'season'], dropna=False)

    for _, group in grouped:
        # Shuffle 
        group = group.sample(frac=1).reset_index(drop=True)

        n = len(group)
        split1 = n // 3
        split2 = 2 * (n // 3) + (n % 3 > 1)  
        
        # Split 
        part1 = group.iloc[:split1]
        part2 = group.iloc[split1:split2]
        
        # 33%
        dataset1.append(part1)
        # 66% 
        dataset2.append(part1)
        dataset2.append(part2)

    # Concatenate the datasets
    dataset1 = pd.concat(dataset1, ignore_index=True)
    dataset2 = pd.concat(dataset2, ignore_index=True)

    # Save to new CSV files

    if(name == 'train_data'):
        TSCV(dataset1, '33%')
        TSCV(dataset1, '66%')
    else:
        dataset1.to_csv(f'33%/{name}_33%.csv', index=False)
        dataset2.to_csv(f'66%/{name}_66%.csv', index=False)
        