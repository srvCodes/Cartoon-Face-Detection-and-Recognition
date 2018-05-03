import os
import re
import cv2
import csv
import numpy as np
from xml.dom import minidom
import pandas as pd

expectedHeight=96
i = 0
dic = {}
dic_check = {}
c = 0
ret = []

dir_path = "./fullCartoonImgsAndXMLs"
dest_path = "/home/saurav/Documents/IIIT-CFW1.0/normalizedFaces_color/"

for filename in os.listdir(dir_path) :
	
	filen, file_extension = os.path.splitext(filename)
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

	cropped_img = img[uly:lry,ulx:lrx]

	newAspectRatio=1.0*expectedHeight/len(cropped_img)
	cropped_img=cv2.resize(cropped_img,(0,0),fx=newAspectRatio,fy=newAspectRatio)
	#print(len(cropped_img[0]))
	if(len(cropped_img[0])<expectedHeight):
		dif=(expectedHeight-len(cropped_img[0]))/2
		cropped_img=cv2.copyMakeBorder(cropped_img,0,0,int(dif),expectedHeight-int(dif)-len(cropped_img[0]),cv2.BORDER_CONSTANT,value=255)
	elif(len(cropped_img[0])>expectedHeight):
		cropped_img=cv2.resize(cropped_img,(expectedHeight,expectedHeight))

	cv2.imwrite(os.path.join(dest_path, filename), cropped_img)
