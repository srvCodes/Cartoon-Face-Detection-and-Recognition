import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os
import re

import tensorflow as tf
import tensorflow.python.platform
from tensorflow.python.platform import gfile
import numpy as np
import pandas as pd
import sklearn
from sklearn import cross_validation, grid_search
from sklearn.metrics import brier_score_loss, accuracy_score, confusion_matrix, classification_report
from sklearn.svm import SVC, LinearSVC
import matplotlib.pyplot as plt
import pickle
from sklearn.calibration import CalibratedClassifierCV
from sklearn import svm
from sklearn.externals import joblib

from sklearn.preprocessing import LabelEncoder
from collections import Counter


model_path = './inception_v3/tensorflow_inception_graph.pb'
images_dir = '/home/saurav/Documents/IIIT-CFW1.0/Without_classes/Class_1_20img/'

images = [images_dir+f for f in os.listdir(images_dir) if re.search('jpg|JPG|jpeg', f)]

def top_n_error(preds, truths, n):
	best_n = np.argsort(preds, axis=1)[:,-n:]
	#ts = np.argmax(truths, axis=1)
	successes = 0
	for i in range(len(truths)):
		if truths[i] in best_n[i,:]:
			successes += 1
	return (1-float(successes)/len(truths))



#print(images)
labels = []

for i in images:
	head, sep, tail = i.partition('/')
	tail = tail.split('0', 1)[0]	
	labels.append(tail)

print(labels)
encoder = LabelEncoder()
encoder.fit(labels)
y = encoder.transform(labels)
print(format(Counter(y)))
'''

def create_graph(model_path):
    """
    create_graph loads the inception model to memory, should be called before
    calling extract_features.
 
    model_path: path to inception model in protobuf form.
    """
    with gfile.FastGFile(model_path, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')

# extract all features from pool layer of InceptionV3
def extract_features(list_images):
	nb_features = 2048
	features = np.empty((len(list_images),nb_features))
	create_graph(model_path)
	with tf.Session() as sess:
		next_to_last_tensor = sess.graph.get_tensor_by_name('pool_3:0')
		for ind, image in enumerate(list_images):
			print('Processing %s...' % (image))
			if not gfile.Exists(image):
				tf.logging.fatal('File does not exist %s', image)
			image_data = gfile.FastGFile(image, 'rb').read()
			predictions = sess.run(next_to_last_tensor,
			{'DecodeJpeg/contents:0': image_data})
			features[ind,:] = np.squeeze(predictions)
	return features

features = extract_features(images)

pickle.dump(features, open('features20_without', 'wb'))
pickle.dump(y, open('y20_without', 'wb'))
'''
features = pickle.load(open('features20_without', 'rb'))
labels = pickle.load(open('y20_without', 'rb'))

X_train, X_test, y_train, y_test = cross_validation.train_test_split(features, labels, 
	test_size=0.2, random_state = 0)

param = [
	{
    	"kernel": ["linear"],
    	"C": [1, 10, 100, 1000]
	},
	{
    	"kernel": ["rbf"],
        "C": [1, 10, 100, 1000],
        "gamma": [1e-2, 1e-3, 1e-4, 1e-5]
	}
    ]

svm = SVC(gamma=0.001, C=100, kernel='rbf', probability=True)
#svm = svm.fit(X_train, y_train)

'''
###### Grid Search For SVM Params ##########
clf = grid_search.GridSearchCV(svm, param, cv = 10, 
	n_jobs = 4, verbose=3)

clf.fit(X_train, y_train)

print(clf.best_params_)

#############################################
'''

final_model = CalibratedClassifierCV(svm, cv=10, method='sigmoid')
final_model = final_model.fit(X_train, y_train)

y_pred_calibrated = final_model.predict_proba(X_test)

y_score = final_model.score(X_test, y_test)
print("============================================")
print("normalizedFaces without bounding box reports:")
print("score:", y_score)

y_pred =  final_model.predict(X_test)
print("accuracy_score for calibrated:")
print(accuracy_score(y_test, y_pred))

print("Top 5 error: ", top_n_error(y_pred_calibrated, y_test, 5)*100)
'''
labels = sorted(list(set(labels)))
print("\nConfusion matrix:")
print("Labels: {0}\n".format(",".join(str(i) for i in labels)))
print(confusion_matrix(y_test, y_pred, labels=labels))
''' 
print("\nClassification report:")
print(classification_report(y_test, y_pred))


def plot_confusion_matrix(y_true,y_pred):
	cm_array = confusion_matrix(y_true,y_pred)
	true_labels = np.unique(y_true)
	pred_labels = np.unique(y_pred)
	plt.imshow(cm_array[:-1,:-1], interpolation='nearest', cmap=plt.cm.Blues)
	plt.title("Confusion matrix", fontsize=16)
	cbar = plt.colorbar(fraction=0.046, pad=0.04)
	cbar.set_label('Number of images', rotation=270, labelpad=30, fontsize=12)
	xtick_marks = np.arange(len(true_labels))
	ytick_marks = np.arange(len(pred_labels))
	plt.xticks(xtick_marks, true_labels, rotation=90)
	plt.yticks(ytick_marks,pred_labels)
	plt.tight_layout()
	plt.ylabel('True label', fontsize=14)
	plt.xlabel('Predicted label', fontsize=14)
	fig_size = plt.rcParams["figure.figsize"]
	fig_size[0] = 12
	fig_size[1] = 12
	plt.rcParams["figure.figsize"] = fig_size
	plt.show()

plot_confusion_matrix(y_test, y_pred)
