import torch
import matplotlib.pyplot as plt
import numpy as np
import idx2numpy
import math

file = './data/train-images.idx3-ubyte'
legend = './data/train-labels.idx1-ubyte'
arr = idx2numpy.convert_from_file(file)
arr2 = idx2numpy.convert_from_file(legend)

f, imgplot = plt.subplots(10,10)

for i in range(10):
    for j in range(10):
        imgplot[i,j].imshow(arr[i*10+j], cmap='gray')
        imgplot[i,j].text(1, 1, arr2[i*10+j], bbox={'facecolor': 'white'})

plt.show()
