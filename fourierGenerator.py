from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.python.keras.utils.data_utils import Sequence
import scipy.stats


class FourierImageDataGenerator():
	def __init__(self, gen, directory, batch_size, img_height, img_width):
		
		self.batch_index = 0
		self.batch_size = batch_size
		self.shape = (img_height, img_width)
		self.generator = gen.flow_from_directory(directory,
                                                        target_size=(
                                                            img_height, img_width),
                                                        class_mode="categorical",
                                                        batch_size=batch_size,
                                                        shuffle=True)
														

														
														
		self.n = self.generator.samples
		
	def reset_index(self):
		self.generator._set_index_array()

	
	def __len__(self):
		return (self.n + self.batch_size - 1) // self.batch_size
	
	def steps_per_epoch(self):
		return self.n // self.batch_size
		
	def __next__(self):
		if self.batch_index == 0:
			self.reset_index()
	
	
		current_index = (self.batch_index * self.batch_size) % self.n
		if self.n > current_index + self.batch_size:
			self.batch_index += 1
		else:
			self.batch_index = 0
			
		# Get a pair of inputs and outputs from iterator
		X, y = self.generator.next()
		
		#For some reason we first have to make X complex, otherwise all complex numbers will vanish when using fft2
		X = X.astype('complex128')
		
		#perform the fourier transform on each image in the batch
		for i in range(len(X)):
			#first test
			X[i] = np.fft.fftshift(np.fft.fft2(X[i]))
			X[i] = np.log(np.abs(X[i])+1)
			
			#rescale to [0,1] if wanted
			X[i] = (X[i] - X[i].min()) / (X[i].max() - X[i].min())
			
			#rescale to [0,255] if wanted (unsure about the np.amin part, I just read that somewhere, I guess X[i].min() should be used)
			#X[i] = X[i] - np.amin(X[i], axis=(0,1), keepdims = True)
			#X[i] = X[i] / X[i].max()
			#X[i] = X[i] * 255
			
		#transfer back to real since all complex numbers are 0, this has to be done som the plot function works etc
		#X = X.astype('float64')
		X = X.real

		return (X, y)
	
	def generate(self):
		while True:
			yield next(self)
			
			
			
		
		
		
		