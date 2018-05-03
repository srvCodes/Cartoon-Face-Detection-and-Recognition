import pandas as pd 
import os
import numpy as np
from sklearn.utils import shuffle
from pandas.io.parsers import read_csv

file1 = "train_own.csv"
file2 = "train_kaggle.csv"

def load(test=False, cols=None):
	allFiles = [file1, file2]

	frame = pd.DataFrame()

	if test == False:
		lis = []

		for file_ in allFiles:
			if file_ == file1:
				df1 = pd.read_csv(file_,index_col=None, header=0)
				lis.append(df1)
				continue
			else:
				df2 = pd.read_csv(file_,index_col=None, header=0)
				lis.append(df2[:2500])
				
			df = pd.concat(lis) 

			print(len(lis[0]))
			print(len(df))
			
			#print(len(frame))
			#print(list(frame.columns.values))

	else:
		df = read_csv(os.path.expanduser('testSetFor100Classes.csv'))

	df['Image'] = df['Image'].apply(lambda im: np.fromstring(im, sep=' '))
	
	if cols:  # get a subset of columns
		df = df[list(cols) + ['Image']]

	#df = df.dropna()
	df = df.fillna(0)

	X = np.vstack(df['Image'].values) / 255.  # scale pixel values to [0, 1]
	X = X.astype(np.float32)
	
	if test == False:
		y = df[df.columns[:-1]].values
		y = (y - 48) / 48  # scale target coordinates to [-1, 1]
		X, y = shuffle(X, y, random_state=42)  # shuffle train data
		y = y.astype(np.float32)

	else:
		y = None
	print(X.shape)
	return X,y

def load2d(test=False, cols=None):
    X, y = load(test=test, cols=cols)
    X = X.reshape(-1, 96, 96, 1)
    
    return X, y
'''
X,y = load2d(test=False)

print(y.max())
'''
