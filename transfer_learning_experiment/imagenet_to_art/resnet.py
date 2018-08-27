import os

import io
import sys
import glob
import keras
import time
import h5py
import random
import gc

import numpy as np 

from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix, roc_auc_score

from keras import __version__
from keras.utils import np_utils
from keras.preprocessing import image
from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import preprocess_input
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
from keras.callbacks import *

from collections import Counter
from PIL import Image

class ResNet(object):
	def __init__(self, hdf5_path, results_path, nb_classes, challenge, tl_mode):

		self.width = 224
		self.height = 224
		self.channels = 3
		self.train_batch_size = 32
		
		self.val_batch_size = 32
		self.test_batch_size = 32

		self.epochs = 100
	    
		self.activation = "relu"
		self.optimizer = SGD(lr = 0.0001, momentum = 0.9)
		self.loss = "categorical_crossentropy"
		self.early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=7, verbose=1, mode='auto')

                self.tl_mode = tl_mode
		self.hdf5_path = hdf5_path + "/"
		self.nb_classes = nb_classes

		self.results_path = results_path + str(challenge) + "/results/" + "ResNet/"
		self.model_path = results_path + str(challenge) + "/models_and_weights/"

		self.make_results_path()
		self.make_model_path()

		self.challenge = challenge

	def load_images(self, name, split):

		f = h5py.File(self.hdf5_path + name, 'r')
		images = list(f[split])

		return(images)

	def load_encodings(self, name, split):
		h5f_labels = h5py.File(self.hdf5_path + name,'r')
		labels = h5f_labels[split][:]
		
		return(labels)

	def my_generator(self, mode):
		
		if mode == "__train":
			X_ = self.X_train
			y_ = self.y_train
			batch_size = self.train_batch_size

		elif mode == "__val":
			X_ = self.X_val
			y_ = self.y_val
			batch_size = self.val_batch_size

		elif mode == "__test":
			X_ = self.X_test
			y_ = self.y_test
			batch_size = self.test_batch_size

		start_batch = 0
		end_batch = start_batch + batch_size
		end_epoch = False

		while True:

			# Returns a random batch indefinitely from X_

			batch = list()
			
			if len(X_) - end_batch < 0:
				end_epoch = True
				start_batch = start_batch - batch_size + 1
				end_batch = end_batch - batch_size + 1

			for imgs in X_[start_batch:end_batch]:
				img = image.load_img(io.BytesIO(imgs), target_size = (self.width, self.height))
				img = image.img_to_array(img)
				img = np.expand_dims(img, axis = 0)	
				img = preprocess_input(img)			  		

				batch.append(img)     

			batch = np.asarray(batch)
			
			X_batch = np.reshape(batch, (batch.shape[0], self.width, self.height, self.channels))
			y_batch = np.asarray([item for item in y_[start_batch:end_batch]])

			yield(X_batch, y_batch) 
			
			start_batch += batch_size
			end_batch += batch_size

			if end_epoch == True:
				start_batch = 0
				end_batch = start_batch + batch_size
				end_epoch = False

	def make_results_path(self):
		if not os.path.exists(self.results_path):
			os.makedirs(self.results_path)

	def make_model_path(self):		
		if not os.path.exists(self.model_path):
			os.makedirs(self.model_path)

	def shape_data(self, images):

		tmp_reshaped_images = list()

		for image in images:
			resized_image = cv2.resize(image, (self.width, self.height)) 
			tmp_reshaped_images.append(resized_image)

		reshaped_images = np.asarray(tmp_reshaped_images)

		X = np.reshape(reshaped_images, (reshaped_images.shape[0], self.width, self.height, self.channels))
		
		return X

	def setup_transfer_learning_model(self, base_model):	
	
	        if self.tl_mode == "off_the_shelf":
		    for layer in base_model.layers[:-2]:
			layer.trainable = False

		        base_model.layers[-1].trainable = True

		        base_model.compile(optimizer = "rmsprop", loss=self.loss, metrics=["accuracy"])

                elif self.tl_mode == "fine_tuning":
                    for layer in base_model.layers:
			layer.trainable = True
		        base_model.compile(optimizer = self.optimizer, loss=self.loss, metrics=["accuracy"])
                    
	def add_layer(self, base_model):

		tmp_output = base_model.output
		tmp_output = GlobalAveragePooling2D()(tmp_output)
		predictions = Dense(self.nb_classes, activation = "softmax")(tmp_output)
		model = Model(input = base_model.input, output = predictions)

		return model

	def train(self):
		base_model = keras.applications.resnet50.ResNet50(weights='imagenet', include_top=False) 
		model = self.add_layer(base_model)

		self.setup_transfer_learning_model(model)

		self.X_train = self.load_images('training_images.hdf5', 'X_train')
		self.y_train = self.load_encodings('training_labels.hdf5', 'y_train')

		self.X_val = self.load_images('validation_images.hdf5', 'X_val')
		self.y_val = self.load_encodings('validation_labels.hdf5', 'y_val')

		self.X_test = self.load_images('testing_images.hdf5', 'X_test')
		self.y_test = self.load_encodings('testing_labels.hdf5', 'y_test')

		tl_history = model.fit_generator(self.my_generator('__train'), steps_per_epoch=len(self.X_train)//self.train_batch_size, nb_epoch=self.epochs, validation_data=self.my_generator('__val'), validation_steps=len(self.X_val)//self.val_batch_size, callbacks = [self.early_stopping])
	
		np.save(self.results_path+"ResNet_transfer_learning_accuracies_full_tuning.npy", tl_history.history["val_acc"])

		tl_score = model.evaluate_generator(self.my_generator('__test'), len(self.X_test)//self.test_batch_size)
		print('Test accuracy via Transfer-Learning:', tl_score[1])

		np.save(self.results_path+"ResNet_test_accuracy.npy", tl_score[1])

		model.save(self.model_path+"ResNet_model.h5")
		model.save_weights(self.model_path + "ResNet_weights.h5")                
