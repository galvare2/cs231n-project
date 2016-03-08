from os import listdir
import os
from scipy import ndimage, misc
import numpy as np

GENUINE = 1
FORGERY = 0
QUESTIONED = -1
IDENTIFICATION = -3
VERIFICATION = -2
GENUINE_FILENAME_LENGTH = 10
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

#load_data_train_without_forge(): This function loads the data for the task for signature verification. Thus,
#the images will be labelled with the identity of the forger
def load_data_train_without_forge():
	X_gen = []
	y_gen = []
	X_forge = []
	y_forge = []
	for filename in listdir(directories['genuine_train']):
		id_num = int(filename[:3])
		add_image(IDENTIFICATION, X_gen, y_gen, id_num, filename, directories['genuine_train'])
	#all of these need to just go in test
	for filename in listdir(directories['forge_train']):
		id_num = -int(filename[4:7])
		add_image(IDENTIFICATION, X_forge, y_forge, id_num, filename, directories['forge_train'])
	for directory in listdir(directories['genuine_test_ref']):
		if directory.startswith('.'): continue
		id_num = int(directory)
		for filename in listdir(os.path.join(directories['genuine_test_ref'], directory)):
			add_image(IDENTIFICATION, X_gen, y_gen, id_num, filename, os.path.join(directories['genuine_test_ref'], directory))
	for directory in listdir(directories['questioned_test']):
		if directory.startswith('.'): continue
		id_num = int(directory)
		for filename in listdir(os.path.join(directories['questioned_test'], directory)):
			if len(filename) == GENUINE_FILENAME_LENGTH:
				add_image(IDENTIFICATION, X_gen, y_gen, id_num, filename, os.path.join(directories['questioned_test'], directory))
			else:
				add_image(IDENTIFICATION, X_forge, y_forge, -id_num, filename, os.path.join(directories['questioned_test'], directory))

	X_gen = np.stack(X_gen, axis=0).astype('float32')
	print "X_gen.shape", X_gen.shape
	y_gen = np.stack(y_gen, axis=0).astype('int32')
	print "y_gen.shape", y_gen.shape
	X_forge = np.stack(X_forge, axis=0).astype('float32')
	print "X_forge.shape", X_forge.shape
	y_forge = np.stack(y_forge, axis=0).astype('int32')
	print "y_forge.shape", y_forge.shape

	np.random.seed(5)
	p = np.random.permutation(len(y_gen))
	X_gen = X_gen[p, :, :, :]
	y_gen = y_gen[p]
	# Apply cutoffs to separate out the data
	train_cutoff_gen = int(len(X_gen) * (1 - TEST_DATA_RATIO - VAL_DATA_RATIO)) # Apply cutoff
	val_cutoff_gen = int(len(X_gen) * (1 - TEST_DATA_RATIO))

	train_cutoff_forge = int(len(X_forge) * (1 - TEST_DATA_RATIO - VAL_DATA_RATIO)) # Apply cutoff
	val_cutoff_forge = int(len(X_forge) * (1 - TEST_DATA_RATIO))

	X_train = X_gen[:train_cutoff_gen]
	print "X_train.shape", X_train.shape
	y_train = y_gen[:train_cutoff_gen]
	print "y_train.shape", y_train.shape
	X_val = np.concatenate((X_gen[train_cutoff_gen:val_cutoff_gen], X_forge[train_cutoff_forge:val_cutoff_forge]))
	print "X_val.shape", X_val.shape
	y_val = np.concatenate((y_gen[train_cutoff_gen:val_cutoff_gen], y_forge[train_cutoff_forge:val_cutoff_forge]))
	print "y_val.shape", y_val.shape
	X_test = np.concatenate((X_gen[val_cutoff_gen:], X_forge[val_cutoff_forge:]))
	print "X_test.shape", X_test.shape
	y_test = np.concatenate((y_gen[val_cutoff_gen:], y_forge[val_cutoff_forge:]))
	print "y_test.shape", y_test.shape
	return X_train, y_train, X_val, y_val, X_test, y_test


def load_data_unseen_test():
	X_train = []
	y_train = []
	X_val_test = []
	y_val_test = []
	for filename in listdir(directories['genuine_train']):
		add_image(VERIFICATION, X_train, y_train, GENUINE, filename, directories['genuine_train'])
	for filename in listdir(directories['forge_train']):
		add_image(VERIFICATION, X_train, y_train, FORGERY, filename, directories['forge_train'])
	for directory in listdir(directories['genuine_test_ref']):
		if directory.startswith('.'): continue
		if int(directory) < UNSEEN_TEST_CUTOFF:
			X_curr = X_train
			y_curr = y_train
		else:
			X_curr = X_val_test
			y_curr = y_val_test
		for filename in listdir(os.path.join(directories['genuine_test_ref'], directory)):
			add_image(VERIFICATION, X_curr, y_curr, QUESTIONED, filename, os.path.join(directories['genuine_test_ref'], directory))
	for directory in listdir(directories['questioned_test']):
		if directory.startswith('.'): continue
		if int(directory) < UNSEEN_TEST_CUTOFF:
			X_curr = X_train
			y_curr = y_train
		else:
			X_curr = X_val_test
			y_curr = y_val_test
		for filename in listdir(os.path.join(directories['questioned_test'], directory)):
			add_image(VERIFICATION, X_curr, y_curr, QUESTIONED, filename, os.path.join(directories['questioned_test'], directory))
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

	return X_train, y_train, X_val, y_val, X_test, y_test

def load_data():
	X_all = []
	y_all = []
	for filename in listdir(directories['genuine_train']):
		add_image(VERIFICATION, X_all, y_all, GENUINE, filename, directories['genuine_train'])
	for filename in listdir(directories['forge_train']):
		add_image(VERIFICATION, X_all, y_all, FORGERY, filename, directories['forge_train'])
	for directory in listdir(directories['genuine_test_ref']):
		if directory.startswith('.'): continue
		for filename in listdir(os.path.join(directories['genuine_test_ref'], directory)):
			add_image(VERIFICATION, X_all, y_all, QUESTIONED, filename, os.path.join(directories['genuine_test_ref'], directory))
	for directory in listdir(directories['questioned_test']):
		if directory.startswith('.'): continue
		for filename in listdir(os.path.join(directories['questioned_test'], directory)):
			add_image(VERIFICATION, X_all, y_all, QUESTIONED, filename, os.path.join(directories['questioned_test'], directory))
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


def add_image(task, X_all, y_all, label, filename, directory):
	if filename.startswith('.'):
		return
	img = ndimage.imread(os.path.join(directory, filename), flatten = False)
	img = np.transpose(img, (2,0,1))
	img2 = np.zeros((3,IMAGE_HEIGHT, IMAGE_WIDTH))
	for i in range(img.shape[0]):
	    im_i = misc.imresize(img[i], (IMAGE_HEIGHT, IMAGE_WIDTH), interp='nearest')
        img2[i] = im_i
	X_all.append(img2)
	if task == VERIFICATION:
		if label is QUESTIONED:
			if len(filename) == GENUINE_FILENAME_LENGTH:
				y_all.append(GENUINE)
			else:
				y_all.append(FORGERY)
		else:
			y_all.append(label)
	else:			
		y_all.append(label)

def main():
	X_train, y_train, X_val, y_val, X_test, y_test = load_data_train_without_forge()

if __name__ == '__main__':
	main()






