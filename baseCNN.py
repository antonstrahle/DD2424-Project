import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, BatchNormalization, GlobalAveragePooling2D
import mixupGenerator as mixupgen
import fourierGenerator as fouriergen
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


IMG_HEIGHT = 224
IMG_WIDTH = 224 
EPOCHS = 3
batch_size  = 100

#Base data
#trainDirectory = "../Data/train"
#validationDirectory = "../Data/valid"
#testDirectory = "../Data/test"

#Fourier
trainDirectory = "../FourierData/train"
validationDirectory = "../FourierData/valid"
testDirectory = "../FourierData/test"

#Used a smaller dataset for testing
#trainDirectory = "../SmallData/train"
#validationDirectory = "../SmallData/valid"
#testDirectory = "../SmallData/test"


classes = os.listdir(trainDirectory)
num_classes = len(classes)


#Base Generators
trainDataGen = ImageDataGenerator(rescale = 1./255.) #rescale as in previous assignment
validDataGen = ImageDataGenerator(rescale = 1./255.) 
testDataGen = ImageDataGenerator(rescale = 1./255.)


#trainDataGen = ImageDataGenerator(rescale = 1./255.,
								  #horizontal_flip = True,
								  #rotation_range = 45,
								  #zoom_range = 0.2,
								  #sheer_range = 0.2)
#validDataGen = ImageDataGenerator(rescale = 1./255.) 
#testDataGen = ImageDataGenerator(rescale = 1./255.)


#====================================================================================											
#with mixup
#====================================================================================
#trainGen = mixupgen.MixupImageDataGenerator(trainDataGen, 
											#trainDirectory,
											#batch_size = batch_size,
											#img_height=IMG_HEIGHT,
											#img_width=IMG_WIDTH,
											#distr = "trunc_norm",
											#params = [0.2, 0.2],
											#majority_vote = 1)
											
#validGen = validDataGen.flow_from_directory(validationDirectory,
											#batch_size = batch_size,
											#class_mode = "categorical",
											#target_size = (IMG_HEIGHT, IMG_WIDTH)) 


#testGen = testDataGen.flow_from_directory(testDirectory,
											#batch_size = batch_size,
											#class_mode = "categorical",
											#target_size = (IMG_HEIGHT, IMG_WIDTH)) 										
#====================================================================================											
#with fourier / IGNORE THIS, PREPROCESSING IS HANDLED BY SAVEING NEW IMAGES, see transform_image_and_save.py
#====================================================================================
#trainGen = fouriergen.FourierImageDataGenerator(trainDataGen, 
											#trainDirectory,
											#batch_size = batch_size,
											#img_height=IMG_HEIGHT,
											#img_width=IMG_WIDTH)											

#validGen = validDataGen.flow_from_directory(validationDirectory,
											#batch_size = batch_size,
											#class_mode = "categorical",
											#target_size = (IMG_HEIGHT, IMG_WIDTH)) 


#testGen = testDataGen.flow_from_directory(testDirectory,
											#batch_size = batch_size,
											#class_mode = "categorical",
											#target_size = (IMG_HEIGHT, IMG_WIDTH)) 
#====================================================================================											
#Standard
#====================================================================================
trainGen = trainDataGen.flow_from_directory(trainDirectory,
											batch_size = batch_size,
											class_mode = "categorical",
											target_size = (IMG_HEIGHT, IMG_WIDTH)) 

#950 img belonging to 190 classes
validGen = validDataGen.flow_from_directory(validationDirectory,
											batch_size = batch_size,
											class_mode = "categorical",
											target_size = (IMG_HEIGHT, IMG_WIDTH)) 

#950 img belonging to 190 classes
testGen = testDataGen.flow_from_directory(testDirectory,
											batch_size = batch_size,
											class_mode = "categorical",
											target_size = (IMG_HEIGHT, IMG_WIDTH)) 

#====================================================================================											
#
#====================================================================================



#General Model. Reliable Model 60% val after a few epochs. Try changing layers to ascending in base 2 (i.e. 16, 32, 64 etc) as well as the stride from 3 to perhaps 1.

model = Sequential([
	Conv2D(64, 1, activation = "relu", input_shape = (IMG_HEIGHT, IMG_WIDTH, 3)),
	MaxPooling2D(2,2),
	BatchNormalization(),
	Dropout(0.4),
	Conv2D(64, 1, activation = "relu"),
	BatchNormalization(),
	Conv2D(64, 1, activation = "relu"),
	MaxPooling2D(2,2),
	BatchNormalization(),
	Dropout(0.4),
	Conv2D(64, 1, activation = "relu"),
	BatchNormalization(),
	Flatten(),
	Dropout(0.5),
	Dense(512, activation = "relu"),
	BatchNormalization(),
	Dense(num_classes, activation = "softmax")
	])


model.compile(optimizer = SGD(lr = 0.01, decay = 1e-6, momentum = 0.9),
				  loss = "categorical_crossentropy",
				  metrics = ["acc"])

model.summary()


###################################################################################
#this is an example on how to plot a batch of images in the training data
#(X, y) = next(trainGen)
#
#def plotImages(images_arr):
#    fig, axes = plt.subplots(1, 5, figsize=(10,10))
#    axes = axes.flatten()
#    for img, ax in zip( images_arr, axes):
#        ax.imshow(img)
#        ax.axis('off')
#    plt.tight_layout()
#    plt.show()
#
#plotImages(X)
###################################################################################


#====================================================================================											
#Use this for mixup
#====================================================================================
#history = model.fit_generator(trainGen.generate(),
							   #steps_per_epoch = trainGen.steps_per_epoch(), #training images / batch size
							   #epochs = EPOCHS,
							   #validation_data = validGen,
							   #validation_steps = 40,
							   #verbose = 1)

#====================================================================================											
#Standard
#====================================================================================
history = model.fit_generator(trainGen,
							   steps_per_epoch = 26769//batch_size, #training images / batch size
							   epochs = EPOCHS,
							   validation_data = validGen,
							   validation_steps = 975//batch_size,
							   verbose = 1)


