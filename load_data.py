from os import listdir
import os
from scipy import ndimage, misc
import numpy as np

GENUINE = 1
FORGERY = 0
QUESTIONED = -1
GENUINE_FILENAME_LENGTH = 10
TEST_DATA_RATIO = .1
VAL_DATA_RATIO = .1
IMAGE_HEIGHT = 224
IMAGE_WIDTH = IMAGE_HEIGHT


directories = {
'genuine_train': '../trainingSet/OfflineSignatures/Dutch/TrainingSet/Offline Genuine',
'forge_train': '../trainingSet/OfflineSignatures/Dutch/TrainingSet/Offline Forgeries',
'genuine_test_ref': '../Testdata_SigComp2011 2/SigComp11-Offlinetestset/Dutch/Reference(646)',
 'questioned_test': '../Testdata_SigComp2011 2/SigComp11-Offlinetestset/Dutch/Questioned(1287)'
}

def load_data():
	X_all = []
	y_all = []
	for filename in listdir(directories['genuine_train']):
		add_image(X_all, y_all, GENUINE, filename, directories['genuine_train'])
	for filename in listdir(directories['forge_train']):
		add_image(X_all, y_all, FORGERY, filename, directories['forge_train'])
	for directory in listdir(directories['genuine_test_ref']):
		if directory.startswith('.'): continue
		for filename in listdir(os.path.join(directories['genuine_test_ref'], directory)):
			add_image(X_all, y_all, QUESTIONED, filename, os.path.join(directories['genuine_test_ref'], directory))
	for directory in listdir(directories['questioned_test']):
		if directory.startswith('.'): continue
		for filename in listdir(os.path.join(directories['questioned_test'], directory)):
			add_image(X_all, y_all, QUESTIONED, filename, os.path.join(directories['questioned_test'], directory))
	#X_all = np.expand_dims(np.stack(X_all, axis=0), axis=1).astype('float32')
	X_all = np.stack(X_all, axis=0).astype('float32')
	y_all = np.stack(y_all, axis=0).astype('int32')
	
	p = np.random.permutation(len(y_all))
	X_all = X_all[p, :, :, :]
	y_all = y_all[p]

	# Apply cutoffs to separate out the data
	train_cutoff = int(len(X_all) * (1 - TEST_DATA_RATIO - VAL_DATA_RATIO)) # Apply cutoff
	val_cutoff = int(len(X_all) * (1 - TEST_DATA_RATIO))
	X_train = X_all[:train_cutoff]
	y_train = y_all[:train_cutoff]
	X_val = X_all[:val_cutoff]
	y_val = y_all[:val_cutoff]
	X_test = X_all[val_cutoff:]
	y_test = y_all[val_cutoff:]
	return X_train, y_train, X_val, y_val, X_test, y_test


def add_image(X_all, y_all, label, filename, directory):
	if filename.startswith('.'):
		return
	#img = ndimage.imread(os.path.join(directory, filename), flatten = True)
	img = ndimage.imread(os.path.join(directory, filename), flatten = False)
        img = np.transpose(img, (2,0,1))
        img2 = np.zeros((3,IMAGE_HEIGHT, IMAGE_WIDTH))
        for i in range(img.shape[0]):
	    im_i = misc.imresize(img[i], (IMAGE_HEIGHT, IMAGE_WIDTH), interp='nearest')
            img2[i] = im_i
	X_all.append(img2)
	if label is QUESTIONED:
		if len(filename) == GENUINE_FILENAME_LENGTH:
			y_all.append(GENUINE)
		else:
			y_all.append(FORGERY)
	else:
		y_all.append(label)

def main():
	X_train, y_train, X_val, y_val, X_test, y_test = load_data()

if __name__ == '__main__':
	main()






