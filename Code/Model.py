# -*- coding: utf-8 -*-
"""

@author: d3evil4
"""

#importing Libraries
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
  except RuntimeError as e:
    print(e)
import numpy as np
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
import cv2
from keras.layers import Flatten
from keras.layers import Dense
from keras.preprocessing import image
import warnings
warnings.filterwarnings('ignore')
from keras.preprocessing.image import ImageDataGenerator


def brightness_adjustment(img):
    # turn the image into the HSV space
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    # return the image int the BGR color space
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)






# initialising the CNN

classifier = Sequential()


# Step 1 Convolution
classifier.add(Convolution2D(32,3,3,input_shape=(64,64,3),activation = 'relu' ))

#step 2 Pooling
classifier.add(MaxPooling2D(pool_size=(2,2)))


# Adding a second comvolution layer
classifier.add(Convolution2D(64,3,3,activation = 'relu' ))

classifier.add(MaxPooling2D(pool_size=(2,2)))



# step3 Flattening
classifier.add(Flatten())

#step4 Full connection
classifier.add(Dense(128,activation = 'relu'))

classifier.add(Dense(1,activation = 'sigmoid'))

# Compiling the CNN
classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])



#Part 2 - Fitting the data

train_datagen = ImageDataGenerator(
	preprocessing_function=brightness_adjustment,
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(preprocessing_function=brightness_adjustment,rescale=1./255)

training_set = train_datagen.flow_from_directory(
        'output/train',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

test_set = test_datagen.flow_from_directory(
        'output/val',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

classifier.fit_generator(training_set,
        epochs=10,
        validation_data=test_set,
        validation_steps=2000)

classifier.save('model.h5') 

