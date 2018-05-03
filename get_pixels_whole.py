import os
import re
import cv2
import csv
import numpy as np
from xml.dom import minidom
import pandas as pd
from sklearn.model_selection import train_test_split
import pickle


expectedHeight=96
i = 0
dic = {}
dic_check = {}
c = 0
ret = []
dir_path = "./fullCartoonImgsAndXMLs"

for filename in os.listdir(dir_path) :
	
	filen, file_extension = os.path.splitext(filename)
	#print(file_extension)
	if file_extension == ".xml" :
		continue
	match = re.match(r"([a-z-]+)([0-9]+).([a-z]+)", filename, re.I)
	if match:
	    item = match.groups()
	    # print(item)
	    if item[0] not in dic:
	    	dic[item[0]] = c + 1
	    	c = c + 1
	else :
		assert False
	if item[2] == "xml" :
		continue
	
	img = cv2.imread(dir_path + "/" + filename,cv2.IMREAD_COLOR)
	obj = minidom.parse(dir_path + "/" + item[0] + item[1] + ".xml")
	obj = obj.getElementsByTagName('zone')
	ulx = max(int(obj[0].attributes['ulx'].value),0)
	uly = max(0,int(obj[0].attributes['uly'].value))
	lrx = max(0,int(obj[0].attributes['lrx'].value))
	lry = max(0,int(obj[0].attributes['lry'].value))
	print(filename)
	# if(filename == "Jay-Z0034.jpeg") :
	# 	print(ulx,uly,lrx,lry,dir_path + "/" + item[0] + item[1] + ".xml")
	# print(type(img))
	cropped_img = img[uly:lry,ulx:lrx]
	#print(cropped_img.shape)

	####################################################
	newAspectRatio=1.0*expectedHeight/len(cropped_img)
	cropped_img=cv2.resize(cropped_img,(0,0),fx=newAspectRatio,fy=newAspectRatio)
	#print(len(cropped_img[0]))
	if(len(cropped_img[0])<expectedHeight):
		dif=(expectedHeight-len(cropped_img[0]))/2
		cropped_img=cv2.copyMakeBorder(cropped_img,0,0,int(dif),expectedHeight-int(dif)-len(cropped_img[0]),cv2.BORDER_CONSTANT,value=255)
	elif(len(cropped_img[0])>expectedHeight):
		cropped_img=cv2.resize(cropped_img,(expectedHeight,expectedHeight))



	####################################################


	dic_check[(cropped_img.shape)] = 1
	# cv2.imshow("cropped",cropped_img)
	# cv2.waitKey(0)
	type(cropped_img)
	#cropped_img = cropped_img.tolist()
	poo = []
	cropped_img = cropped_img.tolist()
	for i in cropped_img:
		poo += i

	poo.append(dic[item[0]])
	ret.append(poo)

#print(ret)
train, test = train_test_split(ret, test_size=0.2)
print(len(train))
print(len(test))


dir_path = "./realFaces"
for filename in os.listdir(dir_path) :
	match = re.match(r"([a-z-]+)([0-9]+).([a-z]+)", filename, re.I)
	if match:
	    item = match.groups()
	    if item[0] not in dic:
	    	assert False 
	else :
		assert False
	
	cropped_img = cv2.imread(dir_path + "/" + filename,cv2.IMREAD_COLOR)

	####################################################
	newAspectRatio=1.0*expectedHeight/len(cropped_img)
	cropped_img=cv2.resize(cropped_img,(0,0),fx=newAspectRatio,fy=newAspectRatio)
	#print(len(cropped_img[0]))
	if(len(cropped_img[0])<expectedHeight):
		dif=(expectedHeight-len(cropped_img[0]))/2
		cropped_img=cv2.copyMakeBorder(cropped_img,0,0,int(dif),expectedHeight-int(dif)-len(cropped_img[0]),cv2.BORDER_CONSTANT,value=255)
	elif(len(cropped_img[0])>expectedHeight):
		cropped_img=cv2.resize(cropped_img,(expectedHeight,expectedHeight))
	dic_check[(cropped_img.shape)] = 1
	poo = []
	#cropped_img = cropped_img.tolist()
	cropped_img = cropped_img.tolist()
	for i in cropped_img:
		#print(i)
		poo += i	
	#poo.append(cropped_img)
	poo.append(dic[item[0]])
	train.append(poo)

################################

#print(len(ret[0]))

#df = pd.DataFrame(data = ret)


#print(train[:10])

#train.to_csv('train.csv')

with open('train_set_all_pixels_color.csv','w') as f:
    writer = csv.writer(f, delimiter ='\t')
    for i in train:
	    writer.writerow(i)

with open('test_set_all_pixels_color.csv','w') as f:
    writer = csv.writer(f, delimiter ='\t')
    for i in test:
	    writer.writerow(i)