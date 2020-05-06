from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.python.keras.utils.data_utils import Sequence
import scipy.stats
import matplotlib.image as mimage



#MAKE A BACKUP OF THE IMAGES BEFORE RUNNING THIS BECAUSE IT WILL OVERWRITE EXISTING IMAGES
def fourier_transform_folder(directory):
	for i in os.listdir(directory):
		folders = []
		folders.append(os.path.join(directory, i))
		for j in folders:
			for k in os.listdir(j):
				filename = os.path.join(j, k)
				#print(filename)
				img=mimage.imread(filename)
				img = img/255
				img = np.fft.fftshift(np.fft.fft2(img))
				img = np.log(np.abs(img)+1)
				
				#rescale to [0,1] if wanted
				img = (img - img.min()) / (img.max() - img.min())
				mimage.imsave(filename, img)


#MAKE A BACKUP OF THE IMAGES BEFORE RUNNING THIS BECAUSE IT WILL OVERWRITE EXISTING IMAGES
fourier_transform_folder("../SmallData/train")
fourier_transform_folder("../SmallData/valid")
fourier_transform_folder("../SmallData/test")