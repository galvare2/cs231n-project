from scipy import ndimage, misc
import numpy as np
import matplotlib.pyplot as plt
import itertools

DATA_LOCATION = "data/"
TEST_DATA_RATIO = 0.1
VAL_DATA_RATIO = 0.1
DATA_EXTENSION = ".jpg"
IMAGE_HEIGHT = 128
IMAGE_WIDTH = 128

def train(data):
	pass

'''
load_data:
Reads in all the data from the data/ folder. Assumes the images are named according to their label
values. Returns a dict data containing all the X and y values cut up into train, val, test sets.
This will include augmented data.
'''

def load_data():
	# Read each image
	data = {}
	X_all = []
	y_all = []
	for i_1 in range(3):
		for i_2 in range(3):
			for i_3 in range(3):
				for i_4 in range(3):
					(imgs, labels) = load_image(i_1, i_2, i_3, i_4, augment=True)
					X_all.extend(imgs)
					y_all.extend(labels)
	
    # Shuffle X_all, y_all in unison to separate similar/augmented data points
	X_all = np.stack(X_all, axis=0)
	y_all = np.stack(y_all, axis=0)
	print X_all.shape
	p = np.random.permutation(len(y_all))
	X_all = X_all[p, :, :, :]
	y_all = y_all[p, :]

    # Apply cutoffs to separate out the data
	train_cutoff = int(len(X_all) * (1 - TEST_DATA_RATIO - VAL_DATA_RATIO)) # Apply cutoff
	val_cutoff = int(len(X_all) * (1 - TEST_DATA_RATIO))
	data['X_train'] = X_all[:train_cutoff]
	data['y_train'] = y_all[:train_cutoff]
	data['X_val'] = X_all[:val_cutoff]
	data['y_val'] = y_all[:val_cutoff]
	data['X_test'] = X_all[val_cutoff:]
	data['y_test'] = y_all[val_cutoff:]
	print data['X_test'].shape
	return data

'''
load_image: Loads an image corresponding to the given label indices. Returns a tuple
(imgs, label) where imgs is a list of images corresponding to the original plus data augmentations,
if augment is set to true.
Rotates images to have the same orientation and then scales them to a set height and width.
'''
def load_image(i_1, i_2, i_3, i_4, augment=True):
	
	# Read image from file. Not included in github repo, must be downloaded from drive.
	img_string = DATA_LOCATION + str(i_1) + str(i_2) + str(i_3) + str(i_4) + DATA_EXTENSION
	img = ndimage.imread(img_string)

	# Rotate so all images have portrait orientation
	if img.shape[1] > img.shape[0]:
		img = np.swapaxes(img, 0, 1)

	# Make all images the same, manageable size
	img = misc.imresize(img, (IMAGE_HEIGHT, IMAGE_WIDTH), interp='nearest')	


	# Make labels. 
	# IMPORTANT NOTE: I'm putting the labels as numpy vectors. We can add more/less structure
	# to this such as changing it to a dictionary if it seems necessary.
	label = (i_1, i_2, i_3, i_4)
	label = np.asarray(label)
	
	# Apply augmentations
	if augment:
		imgs = apply_augmentations(img)
		labels = list(itertools.repeat(label, len(imgs)))
	else:
		imgs = [img]
		labels = [label]
	return (imgs, labels)

'''
apply_augmentations: from an image, returns a list of images which are augmented versions of that
image. The following augmentations are applied:
-color
-brightness
-rotation

'''

def apply_augmentations(img):
	return [img]


'''
model = Sequential()

model.add(Dense(output_dim=64, input_dim=100, init="glorot_uniform"))
model.add(Activation("relu"))
model.add(Dense(output_dim=10, init="glorot_uniform"))
model.add(Activation("softmax")) '''

def main():
	data = load_data()
	train(data)

if __name__ == "__main__":
    main()