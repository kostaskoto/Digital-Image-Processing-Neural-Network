import torch
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import random

from mnist import MNIST

mndata = MNIST('./data')
mndata.gz = True

images, labels = mndata.load_training()

index = random.randrange(0, len(images))  # choose an index ;-)
print(mndata.display(images[index]))

