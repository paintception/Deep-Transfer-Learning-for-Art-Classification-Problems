import pandas as pd 

from resnet import ResNet

from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import train_test_split

from keras.utils import np_utils

from collections import Counter
 
import numpy as np 

import glob
import os 
import csv
import time
import h5py

CHALLENGES = ["dc_type"]

class ExperimentHandler(object):
	def __init__(self, neural_network, dataset_name, metadata_path, jpg_images_path, results_path, dataset_path, tl_mode):
		self.neural_network = neural_network
		self.dataset_name = dataset_name
		self.metadata_path = metadata_path
		self.jpg_images_path = jpg_images_path
                self.tl_mode = tl_mode
		self.dataset_storing_path = dataset_path + dataset_name + "/" 
		self.results_storing_path = results_path + dataset_name + "/"

		self.make_dataset_path() 
		self.make_results_path()

	def make_dataset_path(self):
		if not os.path.exists(self.dataset_storing_path):
			os.makedirs(self.dataset_storing_path)

	def make_results_path(self):
		if not os.path.exists(self.results_storing_path):
			os.makedirs(self.results_storing_path)
	
	def get_metadata(self):
		return(pd.read_csv(self.metadata_path))
		
	def get_images(self):
		images = glob.glob(self.jpg_images_path+"*.jpg")
		images = sorted(images)

		return(images)

	def extract_labels(self, metadata):

		total_labels = list()

		for challenge in CHALLENGES:
			tmp = metadata.loc[:, challenge].tolist()
			total_labels.append(tmp)

		return(total_labels)

	def filter_images_and_labels(self, images, labels):		
			
		to_remove = list()

		for idx, (image, label) in enumerate(zip(images, labels[0])):
			if label == " anoniem" or label == " ":
				to_remove.append(idx)
					
		images = [i for j, i in enumerate(images) if j not in to_remove]
		labels[0] = [i for j, i in enumerate(labels[0]) if j not in to_remove]
	
		return(images, labels)

	def one_hot_encoding(self, total_labels):

		one_hot_encodings = list()
		encoder = LabelEncoder()

		for label in total_labels:

			self.n_labels = len(Counter(label).keys())

			encoder.fit(label)
			encoded_y = encoder.transform(label)

			final_y = np_utils.to_categorical(encoded_y, self.n_labels)

			one_hot_encodings.append(final_y)

		return(one_hot_encodings)

	def store_images_to_hdf5(self, path, images, split):

		f = h5py.File(path)
		dt = h5py.special_dtype(vlen=np.dtype('uint8'))
		dset = f.create_dataset(split, (len(images), ), dtype=dt)
			
		for i in range(0, len(images)):

			filename = images[i]
			fin = open(filename, 'rb')
			binary_data = fin.read()

			dset[i] = np.fromstring(binary_data, dtype='uint8')

	def store_encodings_to_hdf5(self, path, encodings, split):
		f = h5py.File(path)
		dset = f.create_dataset(split, data=encodings)	

	def make_data_splits(self, images, one_hot_encodings):
		
		for challenge in CHALLENGES:

			if not os.path.exists(self.dataset_storing_path+challenge+"/"): 
				os.makedirs(self.dataset_storing_path+challenge+"/")	

				training_images_path = self.dataset_storing_path+challenge+"/"+"training_images.hdf5"
				training_labels_path = self.dataset_storing_path+challenge+"/"+"training_labels.hdf5"
		
				self.hdf5_path = os.path.dirname(training_images_path)

				print("Storing in: ", self.hdf5_path)

				validation_images_path = self.dataset_storing_path+challenge+"/"+"validation_images.hdf5"
				validation_labels_path = self.dataset_storing_path+challenge+"/"+"validation_labels.hdf5"

				testing_images_path = self.dataset_storing_path+challenge+"/"+"testing_images.hdf5"
				testing_labels_path = self.dataset_storing_path+challenge+"/"+"testing_labels.hdf5"

				for labels in one_hot_encodings:
					X_train, X_val, y_train, y_val = train_test_split(images, labels, test_size=0.2, random_state=42)

					self.store_images_to_hdf5(training_images_path, X_train, 'X_train')
					self.store_encodings_to_hdf5(training_labels_path, y_train, 'y_train')

					X_val, X_test, y_val, y_test = train_test_split(X_val, y_val, test_size=0.5 , random_state=42)

					self.store_images_to_hdf5(validation_images_path, X_val, 'X_val')
					self.store_encodings_to_hdf5(validation_labels_path, y_val, 'y_val')

					self.store_images_to_hdf5(testing_images_path, X_test, 'X_test')
					self.store_encodings_to_hdf5(testing_labels_path, y_test, 'y_test')

                    			print("The splits have been created!")
			else:
                                print("The splits are already there!")
				self.hdf5_path = os.path.dirname(self.dataset_storing_path+challenge+"/")

	def run_neural_architecture(self):

		if experiment.neural_network == "ResNet":
			ResNet_net = ResNet(self.hdf5_path, self.results_storing_path, self.n_labels, CHALLENGES[0], self.tl_mode)
			ResNet_net.train()

    	def start_experiment(self):

		images = self.get_images()
		metadata = self.get_metadata()
		
		total_labels = self.extract_labels(metadata)
		
		filtered_data = self.filter_images_and_labels(images, total_labels)
		images = filtered_data[0]
		total_labels = filtered_data[1]

		one_hot_encodings =  self.one_hot_encoding(total_labels)

		self.make_data_splits(images, one_hot_encodings)
		self.run_neural_architecture()

if __name__ == '__main__':
	
	import argparse 

	parser = argparse.ArgumentParser()

	parser.add_argument('--dataset_name', type=str)
	parser.add_argument('--ANN', type=str)
	parser.add_argument('--metadata_path', type=str)
	parser.add_argument('--jpg_images_path', type=str)
	parser.add_argument('--results_path', type=str)
	parser.add_argument('--datasets_path', type=str)
        parser.add_argument('--tl_mode', type=str)

	args = parser.parse_args()

	dataset_name = args.dataset_name
	ANN = args.ANN
	metadata_path = args.metadata_path
	jpg_images_path = args.jpg_images_path
	results_path = args.results_path
	datasets_path = args.datasets_path
        tl_mode = args.tl_mode
	
	experiment = ExperimentHandler(ANN, dataset_name, metadata_path, jpg_images_path, results_path, datasets_path, tl_mode)
	experiment.start_experiment()
