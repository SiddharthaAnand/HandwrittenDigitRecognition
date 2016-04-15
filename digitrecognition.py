
import sys
import pandas as pd
import numpy as np
from sklearn import neighbors
from scipy import sparse

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

	file_name = sys.argv[1]
	data_frame = read_csv(file_name)
	training_values = frequency_of_classes(data_frame)
	
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
	##############################################################

	X, y = divide_train_data(data_frame)

	##############################################################
	# Fit the np array into a classifier 						 #
	# Test the classifier 										 #
	##############################################################

