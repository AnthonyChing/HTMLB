import pandas as pd
import numpy as np
import random
import json

N =  100000 # times to train the model
lrs = [0.001, 0.002, 0.005, 0.01] # fixed learning rate 
def sign(a):
    return 1 if a > 0 else -1
def accuracy(w, x, y):
    correct = 0
    for i in range(len(x)):
        if sign(y[i]) == sign(np.dot(w, x[i])):
            correct += 1
    return float(100 * correct / len(x))
    
def logistic_regression(w, x, y, N, LR): # logistic regression by SGD
    def sigmoid(w, x, y) -> float:
        z = -y * np.dot(w, x)
        return 1 / (1 + np.exp(z))
    
    # stochastic gradient descent
    for times in range(N):
        rand = random.randint(0, len(x) - 1)
        error = sigmoid(w, x[rand], -y[rand])
        for i in range(len(w)):
            w[i] += (LR * error * (y[rand] * x[rand][i]))
    return w

best_acc = 0
best_lr = 0.001
for lr in lrs:
    total_acc = 0
    total_w = []
    for i in range(4): 
        train_data = pd.read_csv(f'train_data_tscv_{i}.csv')
        train_x = train_data.select_dtypes(include = ['float64'])
        train_x = train_x.apply(lambda col: col.fillna(col.median()))
        train_x = train_x.values.tolist()
        train_x = np.delete(train_x, 4, axis = 1)
        
        train_y = np.array(train_data['home_team_win'], dtype = int)
        train_y = [2 * yi - 1 for yi in train_y]
        
        w = np.zeros(len(train_x[0]))
        w = logistic_regression(w, train_x, train_y, N, lr)
        total_w.append(w)
        
        validation = pd.read_csv(f'val_data_tscv_{i}.csv')
        val_x = validation.select_dtypes(include = ['float64'])
        val_x = val_x.apply(lambda col: col.fillna(col.median()))
        val_x = val_x.values.tolist()
        val_x = np.delete(val_x, 4, axis = 1)
        
        val_y = np.array(validation['home_team_win'], dtype = int)
        val_y = [2 * yi - 1 for yi in val_y]
        
        acc = accuracy(w, val_x, val_y)
        total_acc += acc * (i + 1)
    total_acc /= 10
    val_w = np.sum(total_w, axis = 0)
    total_w.clear()
    print(f'Learning rate = {lr}, accuracy = {total_acc}')
    if total_acc > best_acc:
        with open('LR_tscv.json', 'w') as f:
            result = val_w.tolist()
            json.dump(result, f)