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


#Base Generators
trainDataGen = ImageDataGenerator(rescale = 1./255.) #rescale as in previous assignment
validDataGen = ImageDataGenerator(rescale = 1./255.) 
testDataGen = ImageDataGenerator(rescale = 1./255.)

"""
trainDataGen = ImageDataGenerator(rescale = 1./255.,
								  horizontal_flip = True,
								  rotation_range = 45,
								  zoom_range = 0.2,
								  sheer_range = 0.2)
validDataGen = ImageDataGenerator(rescale = 1./255.) 
testDataGen = ImageDataGenerator(rescale = 1./255.)


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


#General Model. Reliable Model 60% val after a few epochs.

testModel = Sequential([
	Conv2D(64, 3, activation = "relu", input_shape = (IMG_HEIGHT, IMG_WIDTH, 3)),
	MaxPooling2D(2,2),
	BatchNormalization(),
	Dropout(0.4),
	Conv2D(64, 3, activation = "relu"),
	BatchNormalization(),
	Conv2D(64, 3, activation = "relu"),
	MaxPooling2D(2,2),
	BatchNormalization(),
	Dropout(0.4),
	Conv2D(64, 3, activation = "relu"),
	BatchNormalization(),
	Flatten(),
	Dropout(0.5),
	Dense(512, activation = "relu"),
	BatchNormalization(),
	Dense(num_classes, activation = "softmax") #Need 190 since we have 190 classes
	])

testModel.compile(optimizer = SGD(lr = 0.01, decay = 1e-6, momentum = 0.9),
				  loss = "categorical_crossentropy",
				  metrics = ["acc"])

#testModel.summary()


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
#####################################
F1 = np.fft.fft2(X[0])
F2 = np.fft.fftshift(F1)
print(F2)
#print(F1.real)




F = np.abs(F2)
F = np.log(F+1)
F = F - np.amin(F, axis=(0,1), keepdims = True)
F = F / F.max()
F = F * 255

plt.imshow(F)
plt.show()

####################################

F1 = np.fft.fft2(X[0])
F2 = np.fft.fftshift(F1)
psd2D = np.abs( F2 )**2
plt.imshow(psd2D)
plt.show()

psd2D = psd2D - np.amin(psd2D, axis=(0,1), keepdims = True)
psd2D = psd2D / psd2D.max()
psd2D = psd2D * 255

plt.imshow(psd2D)
plt.show()


for i in range(len(X)):
	X[i] = np.log10(abs(np.fft.fft2(X[i]))) 

plotImages(X)

