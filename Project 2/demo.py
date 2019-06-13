### Author: Khoong Wei Hao
### Project 2

import imageio
from matplotlib import pyplot as plt
from imcomp import *
from imdecomp import *

#########################
### Image Compression ###
#########################

# import and read greyscale image into a matrix
img_matrix = imageio.imread("fourier.png")

# compression ratio
r = 10

# return coefficient matrix C, the compressed image
C = imcomp(img_matrix, r)

###########################
### Image Decompression ###
###########################

# return image J, the reconstructed image
J = imdecomp(C)

plt.figure(1)
plt.subplot(211)
plt.imshow(img_matrix, cmap="gray")
plt.subplot(212)
plt.imshow(J, cmap="gray")
plt.show()
