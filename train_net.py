from keras.models import Sequential
from keras.layers.core import Dense, Activation
from scipy import ndimage
import numpy as np

TEST_DATA_RATIO = 0.1
VAL_DATA_RATIO = 0.1

def load_data():
	data = {}
	for i_1 in range(3):
		for i_2 in range(3):
			for i_3 in range(3):
				for i_4 in range(3):
					(img, label) = load_image(i_1, i_2, i_3, i_4)
					X_all.append(img)
					y_all.append(label)
	train_cutoff = int(len(X_all) * (1 - TEST_DATA_RATIO - VAL_DATA_RATIO)) # Apply cutoff
	val_cutoff = int(len(X_all) * (1 - TEST_DATA_RATIO))
	data['X_train'] = X_all[:train_cutoff]
	data['y_train'] = y_all[:train_cutoff]
	data['X_val'] = X_all[:val_cutoff]
	data['y_val'] = y_all[:val_cutoff]
	data['X_test'] = X_all[val_cutoff:]
	data['y_test'] = y_all[]
	return data

def load_image(i_1, i_2, i_3, i_4):
	img_string = str(i_1) + str(i_2) + str(i_3) + str(i_4) + ".jpg"
	img = ndimage.imread(img_string)
	print img.shape
	# IMPORTANT NOTE: I'm putting the labels as numpy vectors. We can add more/less structure
	# to this such as changing it to a dictionary if it seems necessary.
	label = (i_1, i_2, i_3, i_4)
	label = np.asarray(label)
	print label.shape
	return (img, label)


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