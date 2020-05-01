#Mixup-generator

from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.python.keras.utils.data_utils import Sequence

class MixupImageDataGenerator():
    def __init__(self, generator, directory, batch_size, img_height, img_width, distr, params):

        self.batch_index = 0
        self.batch_size = batch_size
        self.params = params
        self.distr = distr
        self.shape = (img_height, img_width)
        # First iterator yielding tuples of (x, y)
        self.generator1 = generator.flow_from_directory(directory,
                                                        target_size=(
                                                            img_height, img_width),
                                                        class_mode="categorical",
                                                        batch_size=batch_size,
                                                        shuffle=True)

        # Second iterator yielding tuples of (x, y)
        self.generator2 = generator.flow_from_directory(directory,
                                                        target_size=(
                                                            img_height, img_width),
                                                        class_mode="categorical",
                                                        batch_size=batch_size,
                                                        shuffle=True)

        # Number of images across all classes in image directory.
        self.n = self.generator1.samples

    def reset_index(self):

        self.generator1._set_index_array()
        self.generator2._set_index_array()

    def on_epoch_end(self):
        self.reset_index()

    def reset(self):
        self.batch_index = 0

    def __len__(self):
        # round up
        return (self.n + self.batch_size - 1) // self.batch_size

    def get_steps_per_epoch(self):
        return self.n // self.batch_size

    def __next__(self):

        if self.batch_index == 0:
            self.reset_index()

        current_index = (self.batch_index * self.batch_size) % self.n
        if self.n > current_index + self.batch_size:
            self.batch_index += 1
        else:
            self.batch_index = 0

        # random sample the lambda value from beta distribution.
        if self.distr == "beta":
            l = np.random.beta(self.params, self.params, self.batch_size)
		
		#May add other distributions

        X_l = l.reshape(self.batch_size, 1, 1, 1)
        y_l = l.reshape(self.batch_size, 1)

        # Get a pair of inputs and outputs from two iterators.
        X1, y1 = self.generator1.next()
        X2, y2 = self.generator2.next()

        # Perform the mixup.
        X = X1 * X_l + X2 * (1 - X_l)
        y = y1 * y_l + y2 * (1 - y_l)
        return (X, y)

    #def __iter__(self):
        #while True:
            #yield next(self)
    def generate(self):
        while True:
            yield next(self)

def TestGenerator(generator, directory, batch_size, img_height, img_width):
	generator1 = generator.flow_from_directory(directory,
                                                        target_size=(
                                                            img_height, img_width),
                                                        class_mode="categorical",
                                                        batch_size=batch_size,
                                                        shuffle=True)
														
	generator2 = generator.flow_from_directory(directory,
                                                        target_size=(
                                                            img_height, img_width),
                                                        class_mode="categorical",
                                                        batch_size=batch_size,
                                                        shuffle=True)
	while True:
		l = np.random.beta(0.2, 0.2, batch_size)
		X_l = l.reshape(batch_size, 1, 1, 1)
		y_l = l.reshape(batch_size, 1)
		
		X1, y1 = generator1.next()
		X2, y2 = generator2.next()
	
		X = X1 * X_l + X2 * (1 - X_l)
		y = y1 * y_l + y2 * (1 - y_l)
		
		yield (np.array(X), np.array(y))


