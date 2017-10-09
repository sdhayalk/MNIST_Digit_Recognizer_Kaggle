import numpy as np
import csv

def get_dataset_in_np(dataset_path):
	dataset = []
	first_line_flag = True

	with open(dataset_path) as f:
		dataset_csv_reader = csv.reader(f, delimiter=",")
		for line in dataset_csv_reader:
			if first_line_flag:
				first_line_flag = False
				continue

			temp_line = []
			for element in line:
				temp_line.append(float(element))

			dataset.append(temp_line)

	dataset = np.array(dataset)
	dataset_features = np.array(dataset[:, 1:], dtype='float')
	dataset_labels_temp = np.array(dataset[:, 0], dtype='int')

	dataset_labels = []
	for element in dataset_labels_temp:
		temp = np.zeros(10, dtype='int')	# number of classes is 10
		temp[int(element)] = 1
		dataset_labels.append(temp)

	dataset_labels = np.array(dataset_labels, dtype='int')

	return dataset_features, dataset_labels

def normalize(dataset):
	dataset = dataset / 255
	return dataset

def get_test_dataset_in_np(dataset_path):
	dataset = []
	first_line_flag = True

	with open(dataset_path) as f:
		dataset_csv_reader = csv.reader(f, delimiter=",")
		for line in dataset_csv_reader:
			if first_line_flag:
				first_line_flag = False
				continue

			temp_line = []
			for element in line:
				temp_line.append(float(element))

			dataset.append(temp_line)

	dataset_features = np.array(dataset, dtype='float')

	return dataset_features
