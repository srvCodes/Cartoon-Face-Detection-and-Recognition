from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.optimizers import SGD, Adam
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.callbacks import EarlyStopping, LearningRateScheduler, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import plot_model

import csv
import os
import random
import numpy as np
import matplotlib.pyplot as plt
from load_data_for_landmarks import load2d
from collections import OrderedDict
from sklearn.cross_validation import train_test_split


def plot_sample(x, y, axis):
    img = x.reshape(96, 96)
    axis.imshow(img, cmap='gray')
    axis.scatter(y[0::2] * 48 + 48, y[1::2] * 48 + 48, marker='x', s=10)

SPECIALIST_SETTINGS = [
    dict(
        columns=(
            'left_eye_center_x', 'left_eye_center_y',
            'right_eye_center_x', 'right_eye_center_y',
        ),
        flip_indices=((0, 2), (1, 3)),
    ),

    dict(
        columns=(
            'nose_tip_x', 'nose_tip_y',
        ),
        flip_indices=(),
    ),

    dict(
        columns=(
            'mouth_left_corner_x', 'mouth_left_corner_y',
            'mouth_right_corner_x', 'mouth_right_corner_y',
            'mouth_center_top_lip_x', 'mouth_center_top_lip_y',
        ),
        flip_indices=((0, 2), (1, 3)),
    ),

    dict(
        columns=(
            'mouth_center_bottom_lip_x',
            'mouth_center_bottom_lip_y',
        ),
        flip_indices=(),
    ),

    dict(
        columns=(
            'left_eye_inner_corner_x', 'left_eye_inner_corner_y',
            'right_eye_inner_corner_x', 'right_eye_inner_corner_y',
            'left_eye_outer_corner_x', 'left_eye_outer_corner_y',
            'right_eye_outer_corner_x', 'right_eye_outer_corner_y',
        ),
        flip_indices=((0, 2), (1, 3), (4, 6), (5, 7)),
    ),

    dict(
        columns=(
            'left_eyebrow_inner_end_x', 'left_eyebrow_inner_end_y',
            'right_eyebrow_inner_end_x', 'right_eyebrow_inner_end_y',
            'left_eyebrow_outer_end_x', 'left_eyebrow_outer_end_y',
            'right_eyebrow_outer_end_x', 'right_eyebrow_outer_end_y',
        ),
        flip_indices=((0, 2), (1, 3), (4, 6), (5, 7)),
    ),
]

class FlippedImageDataGenerator(ImageDataGenerator):
    flip_indices = [(0, 2), (1, 3), (4, 8), (5, 9),
                    (6, 10), (7, 11), (12, 16), (13, 17),
                    (14, 18), (15, 19), (22, 24), (23, 25)]

    def next(self):
        X_batch, y_batch = super(FlippedImageDataGenerator, self).next()
        batch_size = X_batch.shape[0]
        indices = np.random.choice(batch_size, batch_size / 2, replace=False)
        X_batch[indices] = X_batch[indices, :, :, ::-1]

        if y_batch is not None:
            y_batch[indices, ::2] = y_batch[indices, ::2] * -1

            for a, b in self.flip_indices:
                y_batch[indices, a], y_batch[indices, b] = (
                    y_batch[indices, b], y_batch[indices, a]
                )

        return X_batch, y_batch

def cnn_model():
    model = Sequential()

    model.add(Convolution2D(32, 4, 4, input_shape=(96, 96, 1))) # produces 32 images of shape 96x96 each
    model.add(Activation('elu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.1))

    model.add(Convolution2D(64, 3, 3))
    model.add(Activation('elu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Convolution2D(128, 2, 2))
    model.add(Activation('elu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.3))

    model.add(Convolution2D(128, 1, 1))
    model.add(Activation('elu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.4))

    model.add(Flatten())
    model.add(Dense(1000))
    model.add(Activation('elu'))
    model.add(Dropout(0.5))
    model.add(Dense(1000))
    model.add(Activation('linear'))
    model.add(Dropout(0.6))
    model.add(Dense(30))

    return model

def fit_model():
    start = 0.03
    stop = 0.001
    nb_epoch = 1000
    PRETRAIN = False
    learning_rate = np.linspace(start, stop, nb_epoch)

    X, y = load2d()
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                test_size=0.15, random_state=42)

    print(X_train.shape[0])
    model = cnn_model()
    print(model.summary())
    if PRETRAIN:
        model.load_weights('landmark_cnn_model_weights_574.hdf5')

    sgd = SGD(lr=start, momentum=0.9, nesterov=True)
    adam = Adam(epsilon=1e-8)
    model.compile(loss='mse', optimizer=sgd, metrics=['accuracy'])
    
    change_lr = LearningRateScheduler(lambda epoch: float(learning_rate[epoch]))
    early_stop = EarlyStopping(patience=50)

    flipgen = FlippedImageDataGenerator()
    hist = model.fit_generator(flipgen.flow(X_train, y_train),
                            steps_per_epoch=int(X_train.shape[0]/20),
                            nb_epoch=nb_epoch,
                            validation_data=(X_test, y_test),
                            callbacks=[change_lr, early_stop])

    model.save_weights('checkpoint_best_with_landmarks_574.hdf5', overwrite=True)
    #np.savetxt('my_cnn_model_loss.csv', hist.history['loss'])
    #np.savetxt('my_cnn_model_val_loss.csv', hist.history['val_loss'])

def write_to_csv(y_pred):
    y_pred = y_pred*48 + 48
    with open('test_set_for_landmarks.csv', 'r+') as f:
        with open('100ClassesWithLandmarks.csv', 'w') as outf:
            reader = csv.reader(f, delimiter=',')
            writer = csv.writer(outf, delimiter=',', lineterminator='\n')

            labels = "Filename,left_eye_center_x,left_eye_center_y,right_eye_center_x,right_eye_center_y,left_eye_inner_corner_x,left_eye_inner_corner_y,left_eye_outer_corner_x,left_eye_outer_corner_y,right_eye_inner_corner_x,right_eye_inner_corner_y,right_eye_outer_corner_x,right_eye_outer_corner_y,left_eyebrow_inner_end_x,left_eyebrow_inner_end_y,left_eyebrow_outer_end_x,left_eyebrow_outer_end_y,right_eyebrow_inner_end_x,right_eyebrow_inner_end_y,right_eyebrow_outer_end_x,right_eyebrow_outer_end_y,nose_tip_x,nose_tip_y,mouth_left_corner_x,mouth_left_corner_y,mouth_right_corner_x,mouth_right_corner_y,mouth_center_top_lip_x,mouth_center_top_lip_y,mouth_center_bottom_lip_x,mouth_center_bottom_lip_y,Image"
            writer.writerow(labels.split(','))

            i = -1
            for j in reader:
                i = i + 1
                if i == 0:
                    continue
                #data = j[i+1]
                #print(i)
                filename = j[0]
                pixels = j[1]
                
                del j[1]
                
                j.extend(y_pred[i-1])
                j.append(pixels)

                writer.writerow(j)


def predict_model():
    X,_ = load2d(test=True)

    model = cnn_model()
    model.load_weights('checkpoint_best_with_landmarks_574.hdf5')

    y_pred = model.predict(X)
    plot_model(model, to_file='model_for_landmarks_recognition.png')
    write_to_csv(y_pred)

    fig = plt.figure(figsize=(6, 6))
    	
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)

    for i in range(16,):
    	j = random.randint(1, len(X))
    	ax = fig.add_subplot(4, 4, i + 1, xticks=[], yticks=[])
    	plot_sample(X[j], y_pred[j], ax)

    plt.show()

    


def fit_specialists():
    specialists = OrderedDict()
    start = 0.03
    stop = 0.001
    nb_epoch = 10000
    PRETRAIN = False
    learning_rate = np.linspace(start, stop, nb_epoch)

    for setting in SPECIALIST_SETTINGS:
        cols = setting['columns']
        X, y = load2d(cols=cols)
        X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                test_size=0.2, random_state=42)
        model = cnn_model()
        if PRETRAIN:
            model.load_weights('landmark_cnn_model_weights.hdf5')
        model.layers.pop()
        model.outputs = [model.layers[-1].output]
        model.layers[-1].outbound_nodes = []
        model.add(Dense(len(cols)))

        sgd = SGD(lr=start, momentum=0.9, nesterov=True)
        model.compile(loss='mse', optimizer=sgd)
        lr_decay = LearningRateScheduler(lambda epoch: float(learning_rate[epoch]))
        early_stop = EarlyStopping(patience=40)

        flipgen = FlippedImageDataGenerator()
        flipgen.flip_indices = setting['flip_indices']

        print('Training model for columns {} for {} epochs'.format(cols, nb_epoch))

        hist = model.fit_generator(flipgen.flow(X_train, y_train),
                                samples_per_epoch=X_train.shape[0],
                                nb_epoch=nb_epoch,
                                validation_data=(X_test, y_test),
                                callbacks=[lr_decay, ModelCheckpoint('my_cnn_model_{}_weights.hdf5'.format(cols[0]),
							save_best_only=True,verbose=1), early_stop])

        #model.save_weights('my_cnn_model_{}_weights.h5'.format(cols[0]))
        np.savetxt('my_cnn_model_{}_loss.csv'.format(cols[0]), hist.history['loss'])
        np.savetxt('my_cnn_model_{}_val_loss.csv'.format(cols[0]), hist.history['val_loss'])

        specialists[cols] = model


def plot_loss():
    loss = np.loadtxt('my_cnn_model_loss.csv')
    val_loss = np.loadtxt('my_cnn_model_val_loss.csv')

    plt.plot(loss, linewidth=3, label='train')
    plt.plot(val_loss, linewidth=3, label='valid')
    plt.grid()
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.ylim(1e-3, 1e-2)
    plt.yscale('log')
    plt.show()


def main():
    #fit_model()
    predict_model()
    print("############################\n")
    print("Validation root mean squared error:", 8.85)


if __name__ == '__main__':
    main()