import numpy as np
import cv2
import os
import random
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout,Activation,Flatten,Conv2D,MaxPool2D

DATADIR = "./DATA_SET"
TYPES = ['C', 'FIST', 'L', 'OKAY', 'PALM', 'PEACE']
training_data = []
X = []
Y = []
IMG_SIZE = 224
# Extracting Data From the folder
# images are 2956 C-208,FIST-550,L-549,OKAY-550,PALM-549,PEACE-550

def create_training_data():
	global X
	global Y
	global IMG_SIZE

	for type_of_image in TYPES:
		path = os.path.join(DATADIR, type_of_image)
		class_num = TYPES.index(type_of_image)
		for img in os.listdir(path):
			try:
				images = cv2.imread(os.path.join(path,img),0)
				new_images = cv2.resize(images, (IMG_SIZE,IMG_SIZE))
				training_data.append([new_images, class_num])

			except Exception as e:
				pass

	random.shuffle(training_data)
	
	for features,label in training_data:
		X.append(features)
		Y.append(label)

	X = np.array(X, dtype = 'float32')
	X = np.stack((X,) * 3, axis=-1)

	Y = np.array(Y).reshape(-1,1)
	X = X/255

	print(X.shape)
	plt.imshow(X[51])
	plt.show()
	print(Y[51])


# def training_model():
# 	global X
# 	global Y
# 	global IMG_SIZE
# 	X = X/255

	model = Sequential()
	model.add(Conv2D(input_shape=(224,224,3),filters=64,kernel_size=(3,3),padding="same", activation="relu"))
	model.add(Conv2D(filters=64,kernel_size=(3,3),padding="same", activation="relu"))
	model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
	model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
	model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
	model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
	model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
	model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
	model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
	model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
	model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
	model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
	model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
	model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
	model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
	model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
	model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
	model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
	model.add(Flatten())
	model.add(Dense(units=4096,activation="relu"))
	model.add(Dense(units=4096,activation="relu"))
	model.add(Dense(units=6, activation="softmax"))

	model.compile(optimizer="Adam", loss="sparse_categorical_crossentropy", metrics=['accuracy'])
	model.fit(X,Y,batch_size = 32,epochs = 5,validation_split = 0.1)

# 	model.save('VGG_16_gesture_recognition_model')
# 	# val_loss,val_acc = model.evaluate(X,y_test)
# 	# print(val_loss)
# 	# print(val_acc)

if __name__ == '__main__':
	create_training_data()
	# training_model()
	


