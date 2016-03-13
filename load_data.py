from os import listdir
import os
from scipy import ndimage, misc
import numpy as np
import matplotlib.pyplot as plt

DISSIMILAR = 0
SIMILAR = 1

GENUINE = 1
FORGERY = 0
QUESTIONED = -1
GENUINE_FILENAME_LENGTH = 10
FORGED_FILENAME_LENGTH = 14
FORGED_FILENAME_TRAIN_SET_LENGTH = 13
TEST_DATA_RATIO = .1
VAL_DATA_RATIO = .1
THRESHOLD = 25
IMAGE_HEIGHT = 224
IMAGE_WIDTH = IMAGE_HEIGHT

UNSEEN_TEST_CUTOFF = 52

directories_dutch = {
'genuine_train': '../trainingSet/OfflineSignatures/Dutch/TrainingSet/Offline Genuine',
'forged_train': '../trainingSet/OfflineSignatures/Dutch/TrainingSet/Offline Forgeries',
'genuine_test_ref': '../Testdata_SigComp2011 2/SigComp11-Offlinetestset/Dutch/Reference(646)',
 'questioned_test': '../Testdata_SigComp2011 2/SigComp11-Offlinetestset/Dutch/Questioned(1287)'
}

directories_chinese = {
'genuine_train': '../trainingSet/OfflineSignatures/Chinese/TrainingSet/Offline Genuine',
'forged_train': '../trainingSet/OfflineSignatures/Chinese/TrainingSet/Offline Forgeries',
'genuine_test_ref': '../Testdata_SigComp2011 2/SigComp11-Offlinetestset/Chinese/Ref(115)',
 'questioned_test': '../Testdata_SigComp2011 2/SigComp11-Offlinetestset/Chinese/Questioned(487)'
}

directories = directories_dutch

def load_data_comparison():
	train_id_dict = {}
	ref_dict = {}

	questioned_dict_val = {}
	questioned_dict_test = {}

	for filename in listdir(directories['genuine_train']):
		if filename.startswith('.'): continue
		id_num = int(filename[:3])
		id_dict = train_id_dict.get(id_num, {'genuine': [], 'forged': []})
		train_id_dict[id_num] = id_dict
		gen_list = id_dict['genuine']
		gen_list.append(os.path.join(directories['genuine_train'], filename))
		train_id_dict[id_num]['genuine'] = gen_list
	#all of these need to just go in test
	for filename in listdir(directories['forged_train']):
		if filename.startswith('.'): continue
		id_num = int(filename[4:7])
		id_dict = train_id_dict.get(id_num, {'genuine': [], 'forged': []})
		train_id_dict[id_num] = id_dict
		forged_list = id_dict['forged']
		forged_list.append(os.path.join(directories['forged_train'], filename))
		train_id_dict[id_num]['forged'] = forged_list

	for directory in listdir(directories['genuine_test_ref']):
		if directory.startswith('.'): continue
		id_num = int(directory)
		ref_dict[id_num] = [os.path.join(directories['genuine_test_ref'], directory, name) for name in listdir(os.path.join(directories['genuine_test_ref'], directory)) if not name.startswith('.')]
	
	val_counter = 0

	for directory in listdir(directories['questioned_test']):
		if val_counter > THRESHOLD:
			questioned_dict = questioned_dict_val
		else:
			questioned_dict = questioned_dict_test
		if directory.startswith('.'): continue
		id_num = int(directory)
		id_dict = questioned_dict.get(id_num, {'genuine': [], 'forged': []})
		questioned_dict[id_num] = id_dict
		for filename in listdir(os.path.join(directories['questioned_test'], directory)):
			if filename.startswith('.'): continue
			if len(filename) == GENUINE_FILENAME_LENGTH:
				gen_list = id_dict['genuine']
				gen_list.append(os.path.join(directories['questioned_test'],directory, filename))
				questioned_dict[id_num]['genuine'] = gen_list
			else:
				forged_list = id_dict['forged']
				forged_list.append(os.path.join(directories['questioned_test'], directory, filename))
				questioned_dict[id_num]['forged'] = forged_list
		val_counter += 1
		print 'questioned_dict', questioned_dict

	X_train, y_train = make_comparison_training_set(train_id_dict)


	X_train = np.stack(X_train, axis=0).astype('float32')

	y_train = np.stack(y_train, axis=0).astype('int32')

#	np.random.seed(5)
#	p = np.random.permutation(len(y_train))
#	X_train = X_train[p, :, :, :]
##	y_train = y_train[p]
	# Apply cutoffs to separate out the data
	# train_cutoff_gen = int(len(X_gen) * (1 - TEST_DATA_RATIO - VAL_DATA_RATIO)) # Apply cutoff
	# val_cutoff_gen = int(len(X_gen) * (1 - TEST_DATA_RATIO))

	# train_cutoff_forge = int(len(X_forge) * (1 - TEST_DATA_RATIO - VAL_DATA_RATIO)) # Apply cutoff
	# val_cutoff_forge = int(len(X_forge) * (1 - TEST_DATA_RATIO))

	# X_train = X_gen[:train_cutoff_gen]
	# y_train = y_gen[:train_cutoff_gen]
	# X_val = np.concatenate((X_gen[train_cutoff_gen:val_cutoff_gen], X_forge[train_cutoff_forge:val_cutoff_forge]))
	# y_val = np.concatenate((y_gen[train_cutoff_gen:val_cutoff_gen], y_forge[train_cutoff_forge:val_cutoff_forge]))
	# X_test = np.concatenate((X_gen[val_cutoff_gen:], X_forge[val_cutoff_forge:]))
	# y_test = np.concatenate((y_gen[val_cutoff_gen:], y_forge[val_cutoff_forge:]))
	# test_cutoff_gen = y_gen[val_cutoff_gen:].shape[0]
	# print "X_train.shape", X_train.shape
	# print "y_train.shape", y_train.shape
	# print "X_val.shape", X_val.shape
	# print "y_val.shape", y_val.shape
	# print "X_test.shape", X_test.shape
	# print "y_test.shape", y_test.shape
	# print "test_cutoff_gen: ", test_cutoff_gen
	print 'len(questioned_dict_test)', len(questioned_dict_test)
	print 'len(questioned_dict_val)', len(questioned_dict_val)
	print 'len(ref_dict)', len(ref_dict)
	return X_train, y_train, ref_dict, questioned_dict_val, questioned_dict_test



def make_comparison_training_set(train_id_dict):
	X_train = []
	y_train = []
	for id_num ,id_dict in train_id_dict.iteritems():
		gen_list = id_dict['genuine']
		forged_list = id_dict['forged']
		for gen in gen_list:
			for forged in forged_list:
				add_comparison_image(gen, forged, X_train, y_train, DISSIMILAR)
			for gen2 in gen_list:
				add_comparison_image(gen, gen2, X_train, y_train, SIMILAR)
                print id_num, " done"                 
	return X_train, y_train


def add_comparison_image(top, bottom, X, y, label):
	top_img = ndimage.imread(top, flatten = False)
	bottom_img = ndimage.imread(bottom, flatten = False)
	top_img = misc.imresize(top_img, (IMAGE_HEIGHT/2, IMAGE_WIDTH), interp='nearest')
	bottom_img = misc.imresize(bottom_img, (IMAGE_HEIGHT/2, IMAGE_WIDTH), interp='nearest')
	img = np.vstack((top_img, bottom_img))
	img2 = np.transpose(img, (2,0,1))
	X.append(img2)
	y.append(label)

def load_data_unseen_separated():
	X_train = []
	y_train = []
	X_val_test = []
	y_val_test = []
	for filename in listdir(directories['genuine_train']):
		add_image(X_val_test, y_val_test, GENUINE, filename, directories['genuine_train'])
	for filename in listdir(directories['forged_train']):
		add_image(X_val_test, y_val_test, FORGERY, filename, directories['forged_train'])
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

def load_data():
	X_all = []
	y_all = []
	for filename in listdir(directories['genuine_train']):
		add_image(X_all, y_all, GENUINE, filename, directories['genuine_train'])
	for filename in listdir(directories['forged_train']):
		add_image(X_all, y_all, FORGERY, filename, directories['forged_train'])
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
    pass

if __name__ == '__main__':
	main()






