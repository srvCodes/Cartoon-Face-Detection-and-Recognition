import os 
import cv2
import csv

path = "/home/saurav/Documents/IIIT-CFW1.0/Without_classes/"
foldername = "Class_2_30img/"

dir_path = path+foldername

ret = []
all_pixels = []


for filename in os.listdir(dir_path) :
	img = cv2.imread(dir_path + "/" + filename,cv2.IMREAD_GRAYSCALE)
	img = img.tolist()
	img = [item for sublist in img for item in sublist]

	ret.append([filename])
	all_pixels.append(img)
	print(len(ret))
	print(len(all_pixels))


with open('testSetFor30Classes.csv','w', newline='') as f:
    writer = csv.writer(f, delimiter =',')
    labels = "Filename,Image"
    writer.writerow(labels.split(','))
    
    for i in range(len(ret)):
	    lis = ret[i]
	    mylist = ' '.join(str(e) for e in all_pixels[i])

	    lis.append(mylist)
	    writer.writerow(lis)
    