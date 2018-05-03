import os
import re
import cv2
import csv
import numpy as np
from xml.dom import minidom
import pandas as pd
from sklearn.model_selection import train_test_split
import pickle


expectedHeight=100
i = 0
dic = {}
dic_check = {}
c = 0
dic2 = {}
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
	    if c == 50 and item[0] not in dic :
	    	continue ;
	    if item[0] not in dic:
	    	dic[item[0]] = c + 1
	    	c = c + 1
	else :
		assert False
	if item[2] == "xml" :
		continue
	
	img = cv2.imread(dir_path + "/" + filename,cv2.IMREAD_GRAYSCALE)
	obj1 = minidom.parse(dir_path + "/" + item[0] + item[1] + ".xml")
	obj = obj1.getElementsByTagName('zone')
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
	
	gen_ele = obj1.getElementsByTagName("p") ;
	if len(gen_ele) == 3 : 

		
		gen_ele = gen_ele[2].firstChild.nodeValue
		print(gen_ele)
		if "Non Frontal" in gen_ele :
			continue	
		poo = []
		cropped_img = cropped_img.tolist()
		for i in cropped_img:
			poo += i
		dic2[dic[item[0]]]=1;
		poo.append(dic[item[0]])
		ret.append(poo)

#print(ret)


dir_path = "./realFaces"
for filename in os.listdir(dir_path) :
	match = re.match(r"([a-z-]+)([0-9]+).([a-z]+)", filename, re.I)
	if match:
	    item = match.groups()
	    if item[0] not in dic:
	    	continue 
	    	assert False 
	else :
		assert False
	
	cropped_img = cv2.imread(dir_path + "/" + filename,cv2.IMREAD_GRAYSCALE)

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
	dic2[dic[item[0]]]=1;
	poo.append(dic[item[0]])
	ret.append(poo)










################################

print(len(ret))

#df = pd.DataFrame(data = ret)
train, test = train_test_split(ret, test_size=0.2)
print(len(train))
print(len(test))

#print(train[:10])

#train.to_csv('train.csv')

with open('train.csv','w') as f:
    writer = csv.writer(f, delimiter ='\t')
    for i in train:
	    writer.writerow(i)

with open('test.csv','w') as f:
    writer = csv.writer(f, delimiter ='\t')
    for i in test:
	    writer.writerow(i)
print(len(dic2))