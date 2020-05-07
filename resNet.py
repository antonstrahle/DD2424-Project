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


trainDirectory = "../Data/train"
validationDirectory = "../Data/valid"
testDirectory = "../Data/test"


#Used a smaller dataset for testing
#trainDirectory = "../SmallData/train"
#validationDirectory = "../SmallData/valid"
#testDirectory = "../SmallData/test"


classes = os.listdir(trainDirectory)
num_classes = len(classes) #Note that the dataset on the virtual machine has 195 classes (due to an update of the data set)


#Base Generators
trainDataGen = ImageDataGenerator(rescale = 1./255.) #rescale as in previous assignment
validDataGen = ImageDataGenerator(rescale = 1./255.) 
testDataGen = ImageDataGenerator(rescale = 1./255.)

#Basic Augemntation
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



resnet50 = tf.keras.applications.ResNet50(input_shape = (IMG_HEIGHT, IMG_WIDTH, 3),
											   include_top = False,
                                               weights='imagenet')

resnet50.trainable = False #we dont alter the pre-trained weights in resnet

model = Sequential([
	resnet50, #resnet50
	GlobalAveragePooling2D(), #pooling
	Dense(num_classes, activation = "softmax") #predictive
	])

model.compile(optimizer = SGD(lr = 0.01, decay = 1e-6, momentum = 0.9),
				  loss = "categorical_crossentropy",
				  metrics = ["acc"])

model.summary()


#====================================================================================											
#Use this for mixup
#====================================================================================
#history = testModel.fit_generator(trainGen.generate(),
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




