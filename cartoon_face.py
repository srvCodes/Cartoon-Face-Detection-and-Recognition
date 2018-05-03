import warnings
warnings.filterwarnings('ignore')

import os
import sys
import random
from tqdm import tqdm
from scipy import misc

import cv2
import matplotlib.pyplot as plt

import tensorflow as tf

import detect_face

root_dir = '/home/saurav/Documents/IIIT-CFW1.0/'

TEST_IMGS_PATH = os.path.join(root_dir, "normalizedFaces_nick/")


minsize = 20
threshold = [ 0.6, 0.7, 0.7 ]  
factor = 0.709 # scale factor

gpu_memory_fraction=1.0



print('Creating networks and loading parameters')

with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            pnet, rnet, onet = detect_face.create_mtcnn(sess, None)

fig = plt.figure(figsize=(6,6))
fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)

choices = ["DalaiLama0080.jpeg",
	"SelenaGomez0033.jpeg",
	"HughJackman0081.jpeg",
	"EmmaWatson0220.jpeg",
	"MattDamon0026.jpeg",
	"JKRowling0002.jpeg",
	"BritneySpears0027.jpeg",
	"JackieChan0032.jpeg",
	"WinstonChurchill0004.jpeg",
	"MarilynMonroe0152.jpeg",
	"MartinLutherKing0020.jpeg",
	"MotherTeresa0111.jpeg",
	"MorganFreeman0010.jpeg",
	"DanielRadicliffe0051.jpeg",
	"LeonardoDiCaprio0090.jpeg",
	"JustinBieber0100.jpeg"]

for i in range(16,):
	#x= random.choice(os.listdir(TEST_IMGS_PATH)) 
	x = choices[i]
	test_image = os.path.join(TEST_IMGS_PATH, x)
	
	print(test_image)
	bgr_image = cv2.imread(test_image)
	rgb_image = bgr_image[:,:,::-1] 

	bounding_boxes, _ = detect_face.detect_face(rgb_image, minsize, pnet, rnet, onet, threshold, factor)

	draw = bgr_image.copy()

	faces_detected = len(bounding_boxes)

	print('Total faces detected ：{}'.format(faces_detected))

	crop_faces=[]


	for face_position in bounding_boxes:
	    face_position=face_position.astype(int)
	    

	    x1 = face_position[0] if face_position[0] > 0 else 0
	    y1 = face_position[1] if face_position[1] > 0 else 0
	    x2 = face_position[2] if face_position[2] > 0 else 0
	    y2 = face_position[3] if face_position[3] > 0 else 0
	    
	    cv2.rectangle(draw, (x1, y1), (x2, y2), (0, 255, 0), 2)
	    
	    crop=bgr_image[y1:y2,x1:x2,]

	ax = fig.add_subplot(4,4, i+1, xticks=[], yticks=[])
	ax.imshow(draw[:,:,::-1], cmap='gray')

plt.show()
