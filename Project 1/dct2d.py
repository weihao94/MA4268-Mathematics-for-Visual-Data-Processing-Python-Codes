### Author: Khoong Wei Hao
### Project 1

import numpy as np
import time

def phi(n):
    if n == 0:
        return (1/2)**(1/2)
    else:
        return 1

def dct2_1d(data):
    N = data.shape[0]
    B = np.zeros([N, N])
    l = np.zeros([N])
    for i in range(N):
        for j in range(N):
            B[i, j] = ((2/N)**(1/2)) * phi(i) * np.cos((j + 1/2) * np.pi * i / N)
    return np.dot(B, data)

def dct2_2d(img_matrix):
    start_time = time.time()
    # note that the image matrix is square, so N is the number of rows/columns
    N = img_matrix.shape[0]
    d = np.empty([N, N], float)
    x = np.empty([N, N], float)
    for i in range(N):
        d[i,:] = dct2_1d(img_matrix[i,:])
    for i in range(N):
        x[:,i] = dct2_1d(d[:,i])
    
    total_time_taken = time.time() - start_time
    print('Total time taken = ' + str(total_time_taken) + 's')
    return x
