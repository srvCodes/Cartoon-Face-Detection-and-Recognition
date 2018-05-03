import pickle
import pandas as pd 
import numpy as np 
import csv
import os
import datetime
import sys
import gc
from keras.models import Sequential, Model
from keras.layers import Dropout, Flatten, Activation, LSTM, Embedding
from keras.layers import Conv1D, Conv2D, MaxPooling2D, GlobalMaxPooling2D, MaxPooling1D
from keras.constraints import maxnorm
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint,  LearningRateScheduler
from keras.layers import Input, Dense 
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import train_test_split
from sklearn.metrics import roc_curve, auc, precision_recall_fscore_support
from collections import Counter

MODE = 'test'
#MODE = 'train'

def find_checkpoint_file(folder):
	checkpoint_file = [f for f in os.listdir(folder) if 'checkpoint' in f]
	if len(checkpoint_file) == 0:
		return []
	modified_time = [os.path.getmtime(f) for f in checkpoint_file]
	return checkpoint_file[np.argmax(modified_time)]

# pickle_file will be closed at this point, preventing your from accessing it any further
X_train = []
y_train = [] 

with open("/home/saurav/Documents/IIIT-CFW1.0/gender_data/train_gender.csv", "r") as input_file:
	read_file = csv.reader(input_file, delimiter='\t')
	for line in read_file:
		poo = []
		poo.append(line[0])
		X_train.append(line[:-1])
		y_train.append(line[-1:])

#print(X_train[0:2])
#print(y_train[0])
#print(type(X_train))
X_train = np.asarray(X_train)

encoder = LabelEncoder()
encoder.fit(y_train)
encoded_Ytrain = encoder.transform(y_train)
print(format(Counter(encoded_Ytrain))) 


X_test = []
y_test = []
with open("/home/saurav/Documents/IIIT-CFW1.0/gender_data/test_gender.csv", "r") as input_file:
	read_file = csv.reader(input_file, delimiter='\t')
	for line in read_file:
		#poo = [] 
		#poo.append(line[0])
		X_test.append(line[:-1])
		y_test.append(line[-1:])

#print(X_test[0])
#print(y_test[0])
X_test = np.asarray(X_test)

# normalize the data from 0-255 to 0.0-1.0
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

X_train = X_train/255.0
X_test = X_test/255.0

# one hot encode outputs
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]


############# 2D Convolutional network ###############

ep = 25
def create_model(X_train, X_test):

	#import pdb;pdb.set_trace()

	visible = Input(shape=(80,80,1))
	conv1 = Conv2D(32, (3,3), activation='relu', padding='same')(visible)
	pool1 = Dropout(0.2)(conv1)
	conv2 = Conv2D(32, (3,3), activation='relu', padding='same')(pool1)
	pool2 = MaxPooling2D(2,2)(conv2)
	conv3 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool2)
	drop2 = Dropout(0.2)(conv3)
	conv4 = Conv2D(64, (3, 3), activation='relu', padding='same')(drop2)
	pool3 = MaxPooling2D(2,2)(conv4)
	conv5 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool3)
	drop3 = Dropout(0.2)(conv5)
	conv6 = Conv2D(128, (3, 3), activation='relu', padding='same')(drop3)
	pool4 = MaxPooling2D(2,2)(conv6)
	flat = Flatten()(pool2)
	drop4 = Dropout(0.2)(flat)
	hidden1 = Dense(1024, activation='relu', kernel_constraint=maxnorm(3))(drop4)
	drop5 = Dropout(0.2)(hidden1)
	hidden2 = Dense(512, activation='relu', kernel_constraint=maxnorm(3))(drop5)
	drop6 = Dropout(0.2)(hidden2)
	output = Dense(num_classes, activation='softmax')(drop6)

	model = Model(inputs = visible, outputs = output)

	epochs = ep 
	lrate = 0.01
	decay = lrate/epochs
	sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
	
	model.compile(loss='categorical_crossentropy',
	              optimizer=sgd,
	              metrics=['accuracy'])
	print(model.summary())
	
	return model

X_tr = np.reshape(X_train, (X_train.shape[0], 80, 80, 1))
X_te = np.reshape(X_test, (X_test.shape[0],80, 80, 1))

X_t, X_v, y_t, y_v = train_test_split(X_tr, y_train, 
						test_size=0.2, random_state=42)
# Image augmentation
datagen = ImageDataGenerator(rotation_range=8, width_shift_range=0.1,
			height_shift_range=0.1, shear_range=0.1,
			zoom_range=0.2, horizontal_flip=True)
train_gen = datagen.flow(X_t, y_t, batch_size=64)

test_datagen = ImageDataGenerator()
test_gen = test_datagen.flow(X_v, y_v, batch_size=64)

model = create_model(X_train, X_test)

if MODE == 'train':
	model.fit_generator(train_gen, steps_per_epoch=X_t.shape[0]/64, epochs=ep, 
		validation_data = test_gen, validation_steps=X_v.shape[0]/64,
		callbacks=[EarlyStopping(patience=5,verbose=1), 
		ModelCheckpoint('',
		save_best_only=True,verbose=1)])

else:
	saved_weights = find_checkpoint_file('.')
	if len(saved_weights) == 0:
		print("Network hasn't been trained yet!")
		sys.exit()
	else:
		model.load_weights('best_gender.hdf5')
		
		#scores = model.evaluate(X_te, y_test, verbose=0)
		print("Loaded saved weights.")
		print("Testing CNN model.. Keep patience !!")
		print("############################")
		print("CNN accuracy: %.2f%%" % (scores[1]*100))
		print("############################")
		'''
		y_pred = model.predict(X_te)
		print(precision_recall_fscore_support(y_test, y_pred))
		'''
