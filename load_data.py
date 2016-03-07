from os import listdir
import os
from scipy import ndimage, misc
import numpy as np

GENUINE = 1
FORGERY = 0
QUESTIONED = -1
GENUINE_FILENAME_LENGTH = 10
FORGED_FILENAME_LENGTH = 14
FORGED_FILENAME_TRAIN_SET_LENGTH = 13
TEST_DATA_RATIO = .1
VAL_DATA_RATIO = .1
IMAGE_HEIGHT = 224
IMAGE_WIDTH = IMAGE_HEIGHT

UNSEEN_TEST_CUTOFF = 52

directories = {
'genuine_train': '../trainingSet/OfflineSignatures/Dutch/TrainingSet/Offline Genuine',
'forge_train': '../trainingSet/OfflineSignatures/Dutch/TrainingSet/Offline Forgeries',
'genuine_test_ref': '../Testdata_SigComp2011 2/SigComp11-Offlinetestset/Dutch/Reference(646)',
 'questioned_test': '../Testdata_SigComp2011 2/SigComp11-Offlinetestset/Dutch/Questioned(1287)'
}

def load_data_unseen_separated():
	X_train = []
	y_train = []
	X_val_test = []
	y_val_test = []
	for filename in listdir(directories['genuine_train']):
		add_image(X_val_test, y_val_test, GENUINE, filename, directories['genuine_train'])
	for filename in listdir(directories['forge_train']):
		add_image(X_val_test, y_val_test, FORGERY, filename, directories['forge_train'])
	for directory in listdir(directories['genuine_test_ref']):
		if directory.startswith('.'): continue
		for filename in listdir(os.path.join(directories['genuine_test_ref'], directory)):
			add_image(X_train, y_train, QUESTIONED, filename, os.path.join(directories['genuine_test_ref'], directory))
	for directory in listdir(directories['questioned_test']):
		if directory.startswith('.'): continue
		for filename in listdir(os.path.join(directories['questioned_test'], directory)):
			add_image(X_train, y_train, QUESTIONED, filename, os.path.join(directories['questioned_test'], directory))
	
	X_train = np.stack(X_train, axis=0).astype('float32')
	y_train = np.stack(y_train, axis=0).astype('int32')
	X_val_test = np.stack(X_val_test, axis=0).astype('float32')
	y_val_test = np.stack(y_val_test, axis=0).astype('int32')

	np.random.seed(5)
	p = np.random.permutation(len(y_train))
	X_train = X_train[p, :, :, :]
	y_train = y_train[p]


	p_2 = np.random.permutation(len(y_val_test))
	X_val_test = X_val_test[p_2, :, :, :]
	y_val_test = y_val_test[p_2]

	cutoff = len(y_val_test) / 2
	X_val = X_val_test[:cutoff]
	y_val = y_val_test[:cutoff]
	X_test = X_val_test[cutoff:]
	y_test = y_val_test[cutoff:]

	print ("X train shape:", X_train.shape)
	print ("y val shape:", y_val.shape)
	print ("y test shape:", y_test.shape)
	print ("y train class skew:", np.bincount(y_train))
	print ("y test class skew:", np.bincount(y_test))

	return X_train, y_train, X_val, y_val, X_test, y_test

def load_data_unseen_test():
	X_train = []
	y_train = []
	X_val_test = []
	y_val_test = []
	for filename in listdir(directories['genuine_train']):
		add_image(X_train, y_train, GENUINE, filename, directories['genuine_train'])
	for filename in listdir(directories['forge_train']):
		add_image(X_train, y_train, FORGERY, filename, directories['forge_train'])
	for directory in listdir(directories['genuine_test_ref']):
		if directory.startswith('.'): continue
		if int(directory) < UNSEEN_TEST_CUTOFF:
			X_curr = X_train
			y_curr = y_train
		else:
			X_curr = X_val_test
			y_curr = y_val_test
		for filename in listdir(os.path.join(directories['genuine_test_ref'], directory)):
			add_image(X_curr, y_curr, QUESTIONED, filename, os.path.join(directories['genuine_test_ref'], directory))
	for directory in listdir(directories['questioned_test']):
		if directory.startswith('.'): continue
		if int(directory) < UNSEEN_TEST_CUTOFF:
			X_curr = X_train
			y_curr = y_train
		else:
			X_curr = X_val_test
			y_curr = y_val_test
		for filename in listdir(os.path.join(directories['questioned_test'], directory)):
			add_image(X_curr, y_curr, QUESTIONED, filename, os.path.join(directories['questioned_test'], directory))
	#X_all = np.expand_dims(np.stack(X_all, axis=0), axis=1).astype('float32')
	X_train = np.stack(X_train, axis=0).astype('float32')
	y_train = np.stack(y_train, axis=0).astype('int32')
	X_val_test = np.stack(X_val_test, axis=0).astype('float32')
	y_val_test = np.stack(y_val_test, axis=0).astype('int32')

	np.random.seed(5)
	p = np.random.permutation(len(y_train))
	X_train = X_train[p, :, :, :]
	y_train = y_train[p]


	p_2 = np.random.permutation(len(y_val_test))
	X_val_test = X_val_test[p_2, :, :, :]
	y_val_test = y_val_test[p_2]

	cutoff = len(y_val_test) / 2
	X_val = X_val_test[:cutoff]
	y_val = y_val_test[:cutoff]
	X_test = X_val_test[cutoff:]
	y_test = y_val_test[cutoff:]

	print ("X train shape:", X_train.shape)
	print ("y val shape:", y_val.shape)
	print ("y test shape:", y_test.shape)
	print ("y train class skew:", np.bincount(y_train))
	print ("y test class skew:", np.bincount(y_test))

	return X_train, y_train, X_val, y_val, X_test, y_test

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
	
        np.random.seed(5)
	p = np.random.permutation(len(y_all))
	X_all = X_all[p, :, :, :]
	y_all = y_all[p]
	# Apply cutoffs to separate out the data
	train_cutoff = int(len(X_all) * (1 - TEST_DATA_RATIO - VAL_DATA_RATIO)) # Apply cutoff
	val_cutoff = int(len(X_all) * (1 - TEST_DATA_RATIO))
	
        X_train = X_all[:train_cutoff]
	y_train = y_all[:train_cutoff]
        X_val = X_all[train_cutoff:val_cutoff]
	y_val = y_all[train_cutoff:val_cutoff]
	X_test = X_all[val_cutoff:]
	y_test = y_all[val_cutoff:]
	return X_train, y_train, X_val, y_val, X_test, y_test

	return X_train, y_train, X_val, y_val, X_test, y_test

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
	
 	np.random.seed(5)
	p = np.random.permutation(len(y_all))
	X_all = X_all[p, :, :, :]
	y_all = y_all[p]
	# Apply cutoffs to separate out the data
	train_cutoff = int(len(X_all) * (1 - TEST_DATA_RATIO - VAL_DATA_RATIO)) # Apply cutoff
	val_cutoff = int(len(X_all) * (1 - TEST_DATA_RATIO))
	X_train = X_all[:train_cutoff]
	y_train = y_all[:train_cutoff]
	X_val = X_all[train_cutoff:val_cutoff]
	y_val = y_all[train_cutoff:val_cutoff]
	X_test = X_all[val_cutoff:]
	y_test = y_all[val_cutoff:]

        print ("y train class skew:", np.bincount(y_train))
        print ("y test class skew:", np.bincount(y_test))

	return X_train, y_train, X_val, y_val, X_test, y_test

def get_forger_id(filename):
	if len(filename) != FORGED_FILENAME_LENGTH: return -1
        if filename[2] != '_': return -1
        f_id = filename[3:7]
	return int(f_id)

def add_image(X, y, label, filename, directory):
	if filename.startswith('.'):
		return
	#img = ndimage.imread(os.path.join(directory, filename), flatten = True)
	img = ndimage.imread(os.path.join(directory, filename), flatten = False)
 	img = np.transpose(img, (2,0,1))
	img2 = np.zeros((3,IMAGE_HEIGHT, IMAGE_WIDTH))
	for i in range(img.shape[0]):
		im_i = misc.imresize(img[i], (IMAGE_HEIGHT, IMAGE_WIDTH), interp='nearest')
    	img2[i] = im_i
	X.append(img2)
	if label is QUESTIONED:
		if len(filename) == GENUINE_FILENAME_LENGTH:
			y.append(GENUINE)
		else:
			y.append(FORGERY)
	else:
		y.append(label)

def main():
	X_train, y_train, X_val, y_val, X_test, y_test = load_data()

if __name__ == '__main__':
	main()






