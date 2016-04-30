'''
bernoulliRBM
'''

from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import BernoulliRBM
from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline
import numpy as np
import time
import cv2

def scale(X, eps=0.001):
	'''
	Scale the data points s.t. the column values are within the range [0, 1].
	'''
	return (X - np.min(X, axis=0)) / (np.max(X, axis=0) + eps)


def nudge(X, y):
	'''
	Make the training data more varied so that the model is less sensitive
	to the raw pixel values and it's changes.
	'''

	# initialize the translations to shift the image one pixel
	# up, down, left, and right, then initialize the new data
	# matrix and targets
	translations = [(0, -1), (0, 1), (-1, 0), (1, 0)]
	data = []
	target = []
 
	# loop over each of the digits
	for (image, label) in zip(X, y):
		# reshape the image from a feature vector of 784 raw
		# pixel intensities to a 28x28 'image'
		image = image.reshape(28, 28)
 
		# loop over the translations
		for (tX, tY) in translations:
			# translate the image
			M = np.float32([[1, 0, tX], [0, 1, tY]])
			trans = cv2.warpAffine(image, M, (28, 28))
 
			# update the list of data and target
			data.append(trans.flatten())
			target.append(label)
 
	# return a tuple of the data matrix and targets
	return (np.array(data), np.array(target))


