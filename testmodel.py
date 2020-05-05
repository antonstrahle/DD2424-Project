import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, BatchNormalization
import mixupGenerator as mixupgen
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


IMG_HEIGHT = 224
IMG_WIDTH = 224 
EPOCHS = 10
batch_size  = 100


trainDirectory = "../Data/train"
validationDirectory = "../Data/valid"
testDirectory = "../Data/test"

"""
#Used a smaller dataset for testing
trainDirectory = "../SmallData/train"
validationDirectory = "../SmallData/valid"
testDirectory = "../SmallData/test"
"""


classes = os.listdir(trainDirectory)
num_classes = len(classes)

trainDataGen = ImageDataGenerator(rescale = 1./255.) #rescale as in previous assignment
validDataGen = ImageDataGenerator(rescale = 1./255.) 
testDataGen = ImageDataGenerator(rescale = 1./255.)

"""
#with mixup
trainGen = mixupgen.MixupImageDataGenerator(trainDataGen, 
											trainDirectory,
											batch_size = batch_size,
											img_height=IMG_HEIGHT,
											img_width=IMG_WIDTH,
											distr = "trunc_norm",
											params = [0.2, 0.2],
											majority_vote = 1)

"""
#Without mixup
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

"""
#General Model. Reliable Model 60% val after a few epochs.

testModel = Sequential([
	Conv2D(64, 3, activation = "relu", input_shape = (IMG_HEIGHT, IMG_WIDTH, 3)),
	MaxPooling2D(2,2),
	BatchNormalization(),
	Dropout(0.5),
	Conv2D(64, 3, activation = "relu"),
	BatchNormalization(),
	Conv2D(64, 3, activation = "relu"),
	MaxPooling2D(2,2),
	BatchNormalization(),
	Dropout(0.5),
	Conv2D(64, 3, activation = "relu"),
	BatchNormalization(),
	Flatten(),
	Dropout(0.5),
	Dense(512, activation = "relu"),
	BatchNormalization(),
	Dense(num_classes, activation = "softmax") #Need 190 since we have 190 classes
	])

testModel.compile(optimizer = SGD(lr = 0.005, decay = 1e-6, momentum = 0.9),
				  loss = "categorical_crossentropy",
				  metrics = ["acc"])

#Try changing LR to 0.01
"""
#IDEK
testModel = Sequential([
	Conv2D(16, 1, activation = "relu", input_shape = (IMG_HEIGHT, IMG_WIDTH, 3)),
	BatchNormalization(),
	Conv2D(32, 3, activation = "relu"),
	BatchNormalization(),
	Conv2D(64, 3, activation = "relu"),
	BatchNormalization(),
	MaxPooling2D(2,2),
	Dropout(0.5),
	Conv2D(64, 3, activation = "relu"),
	BatchNormalization(),
	MaxPooling2D(2,2),
	Dropout(0.5),
	Conv2D(64, 3, activation = "relu"),
	BatchNormalization(),
	MaxPooling2D(2,2),
	Dropout(0.5),
	Conv2D(64, 3, activation = "relu"),
	Flatten(),
	Dropout(0.5),
	Dense(512, activation = "relu"),
	BatchNormalization(),
	Dense(num_classes, activation = "softmax") #Need 190 since we have 190 classes
	])

testModel.compile(optimizer = SGD(lr = 0.01, decay = 1e-6, momentum = 0.9),
				  loss = "categorical_crossentropy",
				  metrics = ["acc"])

testModel.summary()


###################################################################################
#this is an example on how to plot a batch of images in the training data
(X, y) = next(trainGen)

def plotImages(images_arr):
    fig, axes = plt.subplots(1, 5, figsize=(10,10))
    axes = axes.flatten()
    for img, ax in zip( images_arr, axes):
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.show()

plotImages(X)
###################################################################################
"""
history = testModel.fit_generator(trainGen.generate(),
							   steps_per_epoch = trainGen.steps_per_epoch(), #training images / batch size
							   epochs = EPOCHS,
							   validation_data = validGen,
							   validation_steps = 95,
							   verbose = 1)
"""

history = testModel.fit_generator(trainGen,
							   steps_per_epoch = 26769//batch_size, #training images / batch size
							   epochs = EPOCHS,
							   validation_data = validGen,
							   validation_steps = 975//batch_size,
							   verbose = 1)

#does not work yet
"""
acc = history.history['acc']
val_acc = history.history['val_acc']

loss=history.history['loss']
val_loss=history.history['val_loss']

eRange = range(EPOCHS)

#plt.figure(figsize=(8, 8))
#plt.subplot(1, 2, 1)
plt.plot(eRange, acc, label='Training Accuracy')
plt.plot(eRange, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')
plt.show()

#plt.subplot(1, 2, 2)
plt.plot(eRange, loss, label='Training Loss')
plt.plot(eRange, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()
"""
