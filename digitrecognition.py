
import sys
import pandas as pd

def read_csv(file_name):
	'''
	Build a data frame and returns it.
	'''
	data_frame = pd.read_csv(file_name, sep=",")
	return data_frame


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
	### training data 									 #######
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


		