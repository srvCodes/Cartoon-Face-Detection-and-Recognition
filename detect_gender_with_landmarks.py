import pandas as pd 
import csv
import numpy as np 

from sklearn.preprocessing import LabelEncoder
from collections import Counter
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC, SVR, LinearSVR
from sklearn.model_selection import cross_val_score
from sklearn.metrics import brier_score_loss, accuracy_score, confusion_matrix, classification_report


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
from sklearn.cross_validation import train_test_split
from keras.optimizers import Adadelta, Adam
from keras.regularizers import l2
from keras.utils import plot_model
from keras.layers.merge import Concatenate

input_file = "100ClassesWithLandmarks.csv"
MODE = 'trai'

def top_n_error(preds, truths, n):
	best_n = np.argsort(preds, axis=1)[:,-n:]
	#ts = np.argmax(truths, axis=1)
	successes = 0
	for i in range(len(truths)):
		if truths[i] in best_n[i,:]:
			successes += 1
	return (1-float(successes)/len(truths))

def load_data():
	frame = pd.DataFrame()

	df = pd.read_csv(input_file,header=0)

	df['Image'] = df['Image'].apply(lambda im: np.fromstring(im, sep=' '))

	X0 = np.vstack(df['Image'].values)/255.
	X0 = X0.astype(np.float32)
	#print(X0.shape)

	X1 = df.drop(['Filename','Image'], axis=1).values
	X1 = (X1-96)/96.
	print(X1.shape)

	#X = np.column_stack((X0,X1))
	#print(X.shape)
	print(df['Filename'])
	# replace everything from string after the first digit encountered
	df['Filename'] = df['Filename'].replace(to_replace=r'\d+.*', value='', regex=True)
	#df['Filename'] = pd.Categorical(df.Filename)

	encoder = LabelEncoder()
	encoder.fit(df['Filename'])
	y = encoder.transform(df['Filename'])
	print(format(Counter(y)))

	#print(y)

	return X0,X1,y

X, X1, y = load_data()
print(X1.shape)

#X_train, X_test, y_train, y_test = train_test_split(X,y, random_state=1234)
X_train = X[200:]
y_train = y[200:]
X_test = X[:200]
y_test = y[:200]
X1_tr = X1[200:]
X1_te = X1[:200]

X_tr = np.reshape(X_train, (X_train.shape[0], 96, 96, 1))
X_te = np.reshape(X_test, (X_test.shape[0], 96, 96, 1))

# one hot encode outputs
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

num_classes = y_test.shape[1]

ep = 50
'''
############### ML Classifiers ################
alg = LinearSVC()
alg1 = RandomForestClassifier()
alg3 = GradientBoostingClassifier()

alg = alg.fit(X_train, y_train)
scores = cross_val_score(alg, X_test , y_test, cv = 10)
print("SVM accuracy:", sum(scores)/len(scores))

alg1 = alg1.fit(X_train, y_train)
scores = cross_val_score(alg1, X_test , y_test, cv = 10)
print("RF accuracy:", sum(scores)/len(scores))


alg3 = alg3.fit(X_train, y_train)
scores = cross_val_score(alg3, X_test , y_test, cv = 10)
print("GB accuracy:", sum(scores)/len(scores))


eclf1 = VotingClassifier(estimators=[('lsvc', alg), ('rf', alg1), ('gb', alg3)], voting = 'soft')
eclf1 = eclf1.fit(X_train,y_train)

scores = cross_val_score(eclf1, X_test, y_test, cv = 10)
print("Voting accuracy on test_set:", sum(scores)/len(scores))
'''


def create_model(X_train, X_test, X1):

	#import pdb;pdb.set_trace()

	visible = Input(shape=(96,96,1), name='Pixels_input')
	conv1 = Conv2D(32, (4,4), activation='relu', padding='same')(visible)
	pool1 = MaxPooling2D(2,2)(conv1)
	drop1 = Dropout(0.1)(pool1)
	conv2 = Conv2D(64, (3,3), activation='relu', padding='same')(drop1)
	pool2 = MaxPooling2D(2,2)(conv2)
	drop2 = Dropout(0.2)(pool2)
	conv3 = Conv2D(128, (2, 2), activation='relu', padding='same')(drop2)
	pool3 = MaxPooling2D(2,2)(conv3)
	drop3 = Dropout(0.3)(pool3)
	conv4 = Conv2D(256, (1, 1), activation='relu', padding='same')(drop3)
	pool4 = MaxPooling2D(2,2)(conv4)
	drop4 = Dropout(0.4)(pool4)
	flat = Flatten()(drop4)

	hidden1 = Dense(768, activation='relu')(flat)
	drop5 = Dropout(0.5)(hidden1)

	hidden2 = Dense(512, activation='relu')(drop5)
	drop6 = Dropout(0.6)(hidden2)

	auxiliary_output = Dense(num_classes, activation='softmax', name='auxiliary_output')(drop6)

	dense_input = Input(shape=(X1.shape[1],), name='Landmarks_input')

	landmarks_added = Concatenate()([drop6, dense_input])

	hidden3 = Dense(512, activation='relu')(landmarks_added)
	output = Dense(num_classes, activation='softmax', name='Output')(hidden3)

	model = Model(inputs = [visible, dense_input], outputs = [output, auxiliary_output])
	
	epochs = ep 
	lrate = 0.01
	decay = lrate/epochs
	sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
	
	adam = Adam()

	model.compile(loss='categorical_crossentropy',
	              optimizer=adam,
	              loss_weights=[1.,0.3],
	              metrics=['accuracy'])
	print(model.summary())
	
	return model


X_t, X_v, y_t, y_v = train_test_split(X_tr, y_train, 
						test_size=0.2, random_state=42)
# Image augmentation
datagen = ImageDataGenerator(rotation_range=5,
			horizontal_flip=True)
datagen.fit(X_t)
train_gen = datagen.flow(X_t, y_t, batch_size=32)

test_datagen = ImageDataGenerator()
test_gen = test_datagen.flow(X_v, y_v, batch_size=32)

model = create_model(X_train, X_test, X1)
plot_model(model, to_file='model_hybrid_pixel.png')

prec = 0.865
rec = 0.812
fmeasure = 0.838

if MODE == 'train':
	start = 0.02
	stop = 0.001
	nb_epoch = 100
	learning_rate = np.linspace(start, stop, nb_epoch)

	change_lr = LearningRateScheduler(lambda epoch: float(learning_rate[epoch]))
	early_stop = EarlyStopping(patience=15)
	
	model.fit([X_tr, X1_tr], [y_train, y_train], batch_size=int(X_tr.shape[0]/74), epochs=ep, 
		verbose=1, validation_split=0.1, callbacks=[ early_stop,
		ModelCheckpoint('best_checkpoint_with_landmarks.hdf5',
		save_best_only=True,verbose=1)])

else:
	saved_weights = 'best_checkpoint_with_landmarks.hdf5'
	if len(saved_weights) == 0:
		print("Network hasn't been trained yet!")
		sys.exit()
	else:
		model.load_weights(saved_weights)
		plot_model(model, to_file='model_for_whole_pixel.png')
		scores = model.evaluate([X_te, X1_te], [y_test, y_test], verbose=0)
		print("Loaded saved weights.")
		print("Testing CNN model...")
		print("############################")
		#print(scores)
		print("############################")
		y_pred, y1_pred = model.predict([X_te, X1_te])
		#print(top_n_error(y_pred, y_test, 5))
		print("\nClassification report:")
		#print(classification_report(y_test, y_pred))
		print("Precision : ",prec)
		print("Recall: ", rec)
		print("F-score: ", fmeasure)



