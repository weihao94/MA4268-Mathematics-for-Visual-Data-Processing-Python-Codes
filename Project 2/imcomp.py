### Author: Khoong Wei Hao
### Project 2

import numpy as np
import imageio
from scipy.fftpack import dct, idct
from scipy.misc import toimage
import math
import time
from collections import namedtuple
from matplotlib import pyplot as plt

# the following functions phi(n), dct2_1d(data), dct2_2d(img_matrix) are from Project 1, for the 2D DCT
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

#########################################
### Project 2 Functions For imcomp.py ###
#########################################

# function to partition image into 8 X 8 blocks
def partition_img(img, M, N):
    num_partitions_by_row = M/8
    num_partitions_by_col = N/8
    
    img_r = np.array(np.split(img, num_partitions_by_row, axis=0))
    img_rc = np.array(np.split(img_r, num_partitions_by_col, axis=2))
    img_final = np.concatenate(img_rc, axis=0)
    
    return img_final

# function to carry out 2D DCT for each 8X8 patch
def dct2d_8_by_8(patches):
    return [dct2_2d(p) for p in patches]

# function to construct integer coefficient matrix C by concatenating all 8X8 patches
# and converting all entries to 32-bit signed rintegers
def int_coeff_matrix(dct_patches):
    coeff_matrices_3d = np.array(dct_patches)
    
    # note that each 'row' has 256/8=32 8X8 matrices
    reconstructed_matrix_array = []
    for i in range(0,len(coeff_matrices_3d),32):
        # concatenate column-wise
        reconstructed_matrix_array += [np.concatenate(coeff_matrices_3d[i:i+32], axis=1)]

    # stack the 32 'rows' of 8X8 matrices to get 256X256
    reconstructed_matrix = np.vstack(reconstructed_matrix_array)

    # convert all entries to 32-bit signed integers
    return np.rint(reconstructed_matrix).astype(np.int32)

# function to round (1-1/r)MN entries of C with smallest magnitude to zero
def round_smallest_to_zero(matrix, r, M, N):
    num_to_round = int(round((1- 1/r)*M*N))

    # flatten 2D array to 1D so it will be easier to find the smallest k-th value
    flat_matrix_result = matrix.flatten()

    # get smallest num_to_round-th element of coefficient matrix. Note the 0-base indexing
    smallest_val = np.partition(flat_matrix_result, num_to_round-1)[num_to_round]

    # where values of 2D array are lower than the smallest num_to_round-th value
    flags = matrix < smallest_val

    # set smallest values to zero
    matrix[flags] = 0

    return matrix

# performing the image compression with input compression ratio r
def imcomp(I,r):
    # get dimensions of image
    M = I.shape[0]
    N = I.shape[1]

    # get 8X8 patches
    partitions = partition_img(I, M, N)

    # apply 2D DCT on each patch
    dct2d_partitions = dct2d_8_by_8(partitions)

    # convert partitions to integer coefficient matrix
    matrix = int_coeff_matrix(dct2d_partitions)

    # round smallest (1-1/r)MN entries of integer coefficient matrix to zero
    rounded_matrix = round_smallest_to_zero(matrix, r, M, N)

    return rounded_matrix # coefficient matrix C with most entries being zero
