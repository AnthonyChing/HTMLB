import numpy as np

# Create a 3D array with dimensions 5x3x3 filled with zeros

f = open('RF-search-late-stage1-grid-less.txt', 'r')
g = open('RF-search-late-stage2-grid-less.txt', 'r')
array1 = np.zeros((5, 3, 3))
array2 = np.zeros((5, 3, 3))
array3 = np.zeros((5, 3, 3))
array4 = np.zeros((5, 3, 3))
for i in range(5):
    for j in range(3):
        for k in range(3):
            line = f.readline()
            num = float(line.split(' ')[4][:6])
            array1[i][j][k] = f'{(1 - num):.4f}'
            line = g.readline()
            num = float(line.split(' ')[4][:6])
            array2[i][j][k] = f'{(1 - num):.4f}'
for j in range(3):
    for k in range(3):
        for i in range(5):
            print(str(array1[i][j][k]) + '\t' + str(array2[i][j][k]), end= '\t')
        print('')
print('')
h = open('RF-search-tscv-grid-less.txt', 'r')
for i in range(5):
    for j in range(3):
        for k in range(3):
            line = h.readline()
            num = float(line.split(' ')[4][:6])
            array3[i][j][k] = f'{(1 - num):.4f}'
            num = float(line.split(' ')[5][:6])
            array4[i][j][k] = f'{(1 - num):.4f}'
min = 1
for j in range(3):
    for k in range(3):
        for i in range(5):
            print(str(array3[i][j][k]) + '\t' + str(array4[i][j][k]), end= '\t')
            if array4[i][j][k] < min:
                min = array4[i][j][k]
        print('')
print(min)