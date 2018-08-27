import os 

import numpy as np
import tensorflow as tf
import PIL.Image
import keras.backend as K

from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras.preprocessing import image
import numpy as np
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions

from matplotlib import pylab as plt

from keras.models import load_model


RIJKS_MODEL_PATH = "../models/creator/VGG19/"

def show_image(image, grayscale = False, ax=None, title=''):
    if ax is None:
        plt.figure()
    plt.axis('off')
    
    if len(image.shape) == 2 or grayscale == False:
        if len(image.shape) == 3:
            image = np.sum(np.abs(image), axis=2)
            
        vmax = np.percentile(image, 99)
        vmin = np.min(image)

        plt.imshow(image, vmin=vmin, vmax=vmax)
        plt.title(title)
    	
    	plt.show()

    else:
        image = image + 127.5
        image = image.astype('uint8')
        
        plt.imshow(image)
        plt.title(title)
    
    	plt.show()

def load_image(file_path):
    im = PIL.Image.open(file_path)
    im = np.asarray(im)
    
    return im - 127.5

model = VGG19(weights='imagenet')
model.compile(loss='mean_squared_error', optimizer='adam')

img = image.load_img('../images/artist_3.jpg', target_size=(224, 224))
img = np.asarray(img)

show_image(img, grayscale=False)

x = np.expand_dims(img, axis=0)

preds = model.predict(x)
label = np.argmax(preds)

from visual_backprop import VisualBackprop
visual_bprop = VisualBackprop(model)

mask = visual_bprop.get_mask(x[0])
show_image(mask, ax=plt.subplot('121'), title='ImageNet VisualBackProp')

trained_model = load_model(RIJKS_MODEL_PATH + 'creator_VGG19_model.h5')
trained_model.load_weights(RIJKS_MODEL_PATH + 'creator_VGG19_weights.h5')

trained_model.compile(loss='mean_squared_error', optimizer='adam')

visual_bprop = VisualBackprop(trained_model)

mask = visual_bprop.get_mask(x[0])
show_image(mask, ax=plt.subplot('121'), title='RijksNet VisualBackProp')
