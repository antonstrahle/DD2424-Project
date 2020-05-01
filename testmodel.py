import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import RMSprop

inputShape = (150,150) #using 150x150 for testmodel but actual data is 224x224

trainDirectory = "../Data/train"
validationDirectory = "../Data/valid"
testDirectory = "../Data/test"

trainDataGen = ImageDataGenerator(rescale = 1./255.) #rescale as in previous assignment
validDataGen = ImageDataGenerator(rescale = 1./255.) 
testDataGen = ImageDataGenerator(rescale = 1./255.)

#25812 img belonging to 190 classes
trainGen = trainDataGen.flow_from_directory(trainDirectory,
											batch_size = 10,
											class_mode = "categorical",
											target_size = inputShape) 

#950 img belonging to 190 classes
validGen = validDataGen.flow_from_directory(validationDirectory,
											batch_size = 10,
											class_mode = "categorical",
											target_size = inputShape) 

#950 img belonging to 190 classes
testGen = testDataGen.flow_from_directory(testDirectory,
											batch_size = 10,
											class_mode = "categorical",
											target_size = inputShape) 


testModel = tf.ḱeras.models.Sequential([
	tf.python.fkeras.layers.Conv2D(16, (3,3), activation = "relu", input_shape = inputShape),
	tf.keras.layers.MaxPooling2D(2,2),
	tf.keras.layers.Conv2D(32, (3,3), activation = "relu"),
	tf.keras.layers.MaxPooling2D(2,2),
	tf.keras.layers.Conv2D(64, (3,3), activation = "relu"),
	tf.keras.layers.MaxPooling2D(2,2),
	tf.keras.layers.Flatten(),
	tf.keras.layers(1024, activation = "relu"),
	tf.keras.layers.Dense(1, activation = "softmax")
	])

testModel.compile(optimizer = RMSprop(lr = 0.001),
				  loss = "categorical_crossentropy",
				  metrics = ["acc"])

hist = testModel.fit_generator(trainGen,
							   validGen,
							   steps_per_epoch = 100,
							   epochs = 10,
							   validation_steps = 50,
							   verbose = 1)



