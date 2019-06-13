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

# the following functions phi(n), idct2_1d(data), idct2_2d(dct_output) are from Project 1, for the 2D iDCT
def phi(n):
    if n == 0:
        return (1/2)**(1/2)
    else:
        return 1

def idct2_1d(data):
    N = data.shape[0]
    B = np.zeros([N, N])
    l = np.zeros([N])
    for i in range(N):
        for j in range(N):
            B[i, j] = ((2/N)**(1/2)) * phi(j) * np.cos((i + 1/2) * np.pi * j / N)
    return np.dot(B, data)

def idct2_2d(dct_output):
    start_time = time.time()
    # note that the image matrix is square, so N is the number of rows/columns
    N = dct_output.shape[0]
    d = np.empty([N, N], float)
    x = np.empty([N, N], float)
    for i in range(N):
        d[:,i] = idct2_1d(dct_output[:,i])
    for i in range(N):
        x[i,:] = idct2_1d(d[i,:])
    
    total_time_taken = time.time() - start_time
    print('Total time taken = ' + str(total_time_taken) + 's')
    return x

###########################################
### Project 2 Functions For imdecomp.py ###
###########################################

# function to partition image into 8 X 8 blocks
def partition_img(img, M, N):
    num_partitions_by_row = M/8
    num_partitions_by_col = N/8
    
    img_r = np.array(np.split(img, num_partitions_by_row, axis=0))
    img_rc = np.array(np.split(img_r, num_partitions_by_col, axis=2))
    img_final = np.concatenate(img_rc, axis=0)
    
    return img_final

# function to carry out 2D iDCT for each 8X8 patch
def idct2d_8_by_8(patches):
    return [idct2_2d(p) for p in patches]

# function to construct image J by concatenating all 8X8 patches and convert all entries to unsigned 8-bit integers
def reconstruct_image(idct2d_patches):
    coeff_matrices_decomp_3d = np.array(idct2d_patches)

    # note that each 'row' has 256/8=32 8X8 matrices
    concat_matrices_decomp = []
    for i in range(0,len(idct2d_patches),32):
        # concatenate column-wise
        concat_matrices_decomp += [np.concatenate(idct2d_patches[i:i+32], axis=1)]

    # stack the 32 'rows' of 8X8 matrices to get 256X256
    stack_result_decomp = np.vstack(concat_matrices_decomp)

    # convert all entries to unsigned 8-bit integers
    return np.uint8(stack_result_decomp)

# perform image decompression
def imdecomp(C):
    M = C.shape[0]
    N = C.shape[1]
    
    # partition compression output into 8X8 patches
    partitions_comp = partition_img(C, M, N)

    # apply 2D iDCT on each 8X8 patch
    idct2d_partitions = idct2d_8_by_8(partitions_comp)

    # reconstruct image by concatenating all 8X8 patches and converting all entries to 8-bit unsigned integers
    reconstructed_image = reconstruct_image(idct2d_partitions)

    return reconstructed_image
