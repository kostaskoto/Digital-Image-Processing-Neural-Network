import torch
import matplotlib.pyplot as plt
import numpy as np
import idx2numpy

# main program
file = './data/train-images.idx3-ubyte'
legend = './data/train-labels.idx1-ubyte'
imgsTraining = idx2numpy.convert_from_file(file)
labelsTraning = idx2numpy.convert_from_file(legend)

f, imgplot = plt.subplots(1,10)
lstImg = []
lstLabel = []

for i in range(len(labelsTraning)):
    if not labelsTraning[i] in lstLabel:
        lstImg.append(imgsTraining[i])
        lstLabel.append(labelsTraning[i])
    if len(lstImg) == 10:
        break

sort = np.argsort(lstLabel)
lstImg = np.array(lstImg)[sort]
lstLabel = np.array(lstLabel)[sort]

for i in range(len(lstImg)):       
    imgplot[i].imshow(lstImg[i], cmap='gray')
    imgplot[i].text(1, 1, lstLabel[i], bbox={'facecolor': 'white'})

plt.show()
