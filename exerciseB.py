import numpy as np
from sklearn import datasets, svm, metrics
from skimage.feature import hog
from sklearn.model_selection import train_test_split
import idx2numpy
from tabulate import tabulate

# Load the MNIST dataset
fileTraining = './data/train-images.idx3-ubyte'
legendTraining = './data/train-labels.idx1-ubyte'
features_train = idx2numpy.convert_from_file(fileTraining)
labels_train = idx2numpy.convert_from_file(legendTraining)
fileTesting = './data/t10k-images.idx3-ubyte'
legendTesting = './data/t10k-labels.idx1-ubyte'
features_test = idx2numpy.convert_from_file(fileTesting)
labels_test = idx2numpy.convert_from_file(legendTesting)

# Reshape the features
features_train = features_train.reshape((features_train.shape[0], -1))
features_test = features_test.reshape((features_test.shape[0], -1))

# Extract HOG features from the training set
hog_features_train = []
for image in features_train:
    hog_features_train.append(hog(image.reshape((28, 28)), orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1), visualize=False))

# Train the SVM classifier
svm_classifier = svm.SVC(kernel='linear')
svm_classifier.fit(hog_features_train, labels_train)

# Extract HOG features from the testing set
hog_features_test = []
for image in features_test:
    hog_features_test.append(hog(image.reshape((28, 28)), orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1), visualize=False))

# Predict labels for the testing set
predicted_labels = svm_classifier.predict(hog_features_test)

# Print classification report
print("Classification Report:")
print(metrics.classification_report(labels_test, predicted_labels))

# Create confusion matrix
confusion_matrix = metrics.confusion_matrix(labels_test, predicted_labels)
print("Confusion Matrix:")
print(tabulate(confusion_matrix, headers=[0,1,2,3,4,5,6,7,8,9], tablefmt='orgtbl'))