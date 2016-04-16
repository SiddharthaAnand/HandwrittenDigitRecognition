
import csv	
import sys
import pandas as pd
import numpy as np
from sklearn import svm
from scipy import sparse
from sklearn.ensemble import RandomForestClassifier

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

	print 'Class\t', '#training data'
	for i in range(len(training_values)):
		print i, '\t', training_values[i]

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
	print "X_test ", len(X_test[0])

	##############################################################
	# Fit the np array into a classifier 						 #
	# Test the classifier 										 #
	##############################################################

	classifier = RandomForestClassifier()
	classifier.fit(X_train, y_train)
	prediction = classifier.predict(X_test)

	##############################################################
	#Create file of test data class to upload                    #
	##############################################################

	c = 0
	writer = csv.writer(open("results.csv", "w"))

	if c == 0:
		writer.writerow(["ImageId", "Label"])

	for digit_class in prediction:
		c += 1
		writer.writerow([c, digit_class])
		

	print "##########prediction over###############"
	print "#Result stored in results.csv          #"
	print "########################################"
