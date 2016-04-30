
import cv2
import time
import csv	
import sys
import pandas as pd
import numpy as np
from sklearn import neighbors
from PIL import Image
from scipy import sparse
from sklearn.ensemble import RandomForestClassifier
from sklearn import grid_search
from sklearn.neural_network import BernoulliRBM
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn import linear_model
from sklearn.externals import joblib
from skimage.feature import hog
from sklearn.svm import LinearSVC
import bernoulliRBM
import logistic_regression
import preprocessing

def read_csv(file_name):
	'''
	Build a data frame and returns it.
	'''
	data_frame = pd.read_csv(file_name, sep=",")
	return data_frame

def divide_train_data(data_frame):
	'''
	Take data frame and divide into training data format.
	X = training examples, every row is a single example.
	y = corresponding class denoted by the first column.
	'''

	X = []
	y = []

	for row_index in range(len(data_frame)):
		X.append(np.array(data_frame.iloc[row_index, 1: ]))
		y.extend([data_frame.iloc[row_index]['label']])

	return X, y

def test_data_numpy_array(test_data_frame):
	'''
	This method takes the test data frame consisting of features
	and converts it into numpy array to test using sklearn.	
'''
	X = []
	for row_index in range(len(test_data_frame)):
		X.append(np.array(test_data_frame.iloc[row_index, 0: ]))

	return X

def frequency_of_classes(data_frame):
	'''
	Count the frequency of different classes in the training data.
	'''

	classes = [0] * 10
	for row_index in range(len(data_frame)):
		classes[data_frame.loc[row_index]['label']] += 1

	return classes

if __name__ == '__main__':
	
	############################################################
	### Read the csv file from the command line          #######
	### Find the frequency of the classes present in the #######
	### training data 						             #######
	############################################################

	train_file_name = sys.argv[1]
	test_file_name = sys.argv[2]
	data_frame = read_csv(train_file_name)
	training_values = frequency_of_classes(data_frame)
	test_data = []

	############################################################
	#Print the frequency of the 9 classes in the training data #
	############################################################

	'''
	print 'Class\t', '#training data'
	for i in range(len(training_values)):
		print i, '\t', training_values[i]
	'''
	##############################################################
	#     Convert data frame into numpy array for sklearn        #
	# X - extract column 1 to 784 as training features(pixels)   #
	# y - class of the examples									 #
	# test_data_frame - data frame for test data                 #
	# X_test - numpy array of test data                          #
	##############################################################

	X_train, y_train = divide_train_data(data_frame)
	test_data_frame = read_csv(test_file_name)
	X_test = test_data_numpy_array(test_data_frame)
	X_test = np.array(X_test)
	#print "X_test ", X_test
	prediction = []

	'''
	##############################################################
	# Fit the np array into a Random Forest classifier			 #
	# Test the classifier 										 #
	##############################################################

	classifier = RandomForestClassifier()
	classifier.fit(X_train, y_train)
	prediction = classifier.predict(X_test)
	
	'''
	'''
	##############################################################
	# Fit the np array into a Nearest neighbor classifier		 #
	# Test the classifier 										 #
	##############################################################

	classifier = neighbors.KNeighborsClassifier(n_neighbors=10, weights='distance', n_jobs=4)
	classifier.fit(X_train, y_train)
	prediction = classifier.predict(X_test)
	'''

	'''
	#############################################################
	# Using Grid Search 										#
	#############################################################


	X_train = bernoulliRBM.scale(X_train)
	X_train, y_train = bernoulliRBM.nudge(X_train, y_train)
	X_train = preprocessing.sparse_matrix(X_train)
	rbm = BernoulliRBM()
	logistic = linear_model.LogisticRegression()
	classifier = Pipeline([("rbm", rbm), ("logistic", logistic)])

	print "Searching RBM + Logistic Regression"
	params = {
		"rbm__learning_rate": [0.1, 0.01, 0.001],
		"rbm__n_iter": [20, 40, 80],
		"rbm__n_components": [50, 100, 200],
		"logistic__C": [1.0, 10.0, 100.0]}
 
	# perform a grid search over the parameter
	start = time.time()
	gs = GridSearchCV(classifier, params, n_jobs = 4, verbose = 1)
	gs.fit(X_train, y_train)
 
	# print diagnostic information to the user and grab the
	# best model
	print "\ndone in %0.3fs" % (time.time() - start)
	print "best score: %0.3f" % (gs.best_score_)
	print "RBM + LOGISTIC REGRESSION PARAMETERS"
	bestParams = gs.best_estimator_.get_params()
 
	# loop over the parameters and print each of them out
	# so they can be manually set
	for p in sorted(params.keys()):
		print "\t %s: %f" % (p, bestParams[p])
	
	
	prediction = logistic_regression.logistic_regression(X_train, y_train, X_test)
	'''

	##############################################################
	# Linear SVM with HOG features								 #
	##############################################################

	list_hog_fd =[]
	for feature in X_train:
		fd = hog(feature.reshape(28, 28), orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1), visualise=False)
		list_hog_fd.append(fd)

	X_train = np.array(list_hog_fd, 'float64')

	clf = LinearSVC()
	start_time = time.time()
	print "Training "
	clf.fit(X_train, y_train)
	print " Total time : ", time.time() - start_time

	#Pickling :-> joblib.dump(clf, 'digits_clf.pkl', compress=3)

	list_hog_fd =[]
	count = 0
	for i in X_test:
		count += 1
		buff = []
		data = i.reshape(28, 28)
		img = Image.fromarray(data, 'RGB')
		img.save('test.png')

		im = cv2.imread("test.png")
		im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
		im_gray = cv2.GaussianBlur(im_gray, (5, 5), 0)
		ret, im_th = cv2.threshold(im_gray, 90, 255, cv2.THRESH_BINARY_INV)
		ctrs, hier = cv2.findContours(im_th.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
		rects = [cv2.boundingRect(ctr) for ctr in ctrs]

		for rect in rects:
			cv2.rectangle(im, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 255, 0), 3)
			
		if count % 100 == 0:
			print count, " images converted " 
		#new_X_test = np.array(data)
		fd = hog(i.reshape(28, 28), orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1), visualise=False)
		list_hog_fd.append(fd)

		#im_gray = cv2.cvtColor(np.array(X_test), cv2.COLOR_BGR2GRAY)
		
	X_test = list_hog_fd
	prediction = clf.predict(X_test)
	##############################################################
	#Create csv file of test data class to upload                #
	#results.csv file format									 #
	# ImageId Label  											 #
	# 1       0 												 #
	# ...     ... 												 #
 	##############################################################

	c = 0
	writer = csv.writer(open("results_svm_hog.csv", "w"))

	if c == 0:
		writer.writerow(["ImageId", "Label"])

	for digit_class in prediction:
		c += 1
		writer.writerow([c, digit_class])
		
