### Author: Khoong Wei Hao
### Project 1

import numpy as np
import time
import imageio
from matplotlib import pyplot as plt
from dct2d import *
from idct2d import *

# import and read greyscale image into a matrix
img_matrix = imageio.imread("fourier.png")

dct_output = dct2_2d(img_matrix)

idct_output = idct2_2d(dct_output)

plt.figure(1)
plt.subplot(211)
plt.imshow(dct_output, cmap="gray")
plt.subplot(212)
plt.imshow(idct_output, cmap="gray")
plt.show()
