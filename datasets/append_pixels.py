import csv
import os
import cv2
import fileinput
import pandas as pd 

dir_path = "/home/saurav/Documents/IIIT-CFW1.0/normalizedFaces"
csv_dir = "landmarks.csv"

all_pixels = []

with open(csv_dir, 'r+', encoding='utf-8') as f:
	reader = csv.reader(f, delimiter=',')
	c = 1
	for data in reader:
		'''
		if c > 1: 
			continue
		c += 1
		'''
	
		filename = data[0].split(',')[0]
		#print(filename)
		

		for file in os.listdir(dir_path) :
			if file == filename :
				filen, file_extension = os.path.splitext(filename)

				img = cv2.imread(dir_path + "/" + filename,cv2.IMREAD_GRAYSCALE)
				img= img.tolist()

				#print(img)
				
				img = [item for sublist in img for item in sublist]
				
				all_pixels.append(img)
				
						
i = 0
with open(csv_dir, 'r+', encoding='utf-8') as f:
	with open("train.csv", 'w', newline='') as outf:
		reader = csv.reader(f, delimiter=',')
		writer = csv.writer(outf, delimiter=',', lineterminator='\n')
		labels = "left_eye_center_x,left_eye_center_y,right_eye_center_x,right_eye_center_y,left_eye_inner_corner_x,left_eye_inner_corner_y,left_eye_outer_corner_x,left_eye_outer_corner_y,right_eye_inner_corner_x,right_eye_inner_corner_y,right_eye_outer_corner_x,right_eye_outer_corner_y,left_eyebrow_inner_end_x,left_eyebrow_inner_end_y,left_eyebrow_outer_end_x,left_eyebrow_outer_end_y,right_eyebrow_inner_end_x,right_eyebrow_inner_end_y,right_eyebrow_outer_end_x,right_eyebrow_outer_end_y,nose_tip_x,nose_tip_y,mouth_left_corner_x,mouth_left_corner_y,mouth_right_corner_x,mouth_right_corner_y,mouth_center_top_lip_x,mouth_center_top_lip_y,mouth_center_bottom_lip_x,mouth_center_bottom_lip_y,Image"
		writer.writerow(labels.split(','))
	
		for j in reader:
			mylist = ' '.join(str(e) for e in all_pixels[i])
			i = i + 1
			lis = j[0].split(',')
			del lis[0]
			'''
			for k in range(len(lis)) : 
				if k%2 == 1 and lis[k] != '': 
					lis[k] = str(int(lis[k]) - 121)
			'''
			lis.append(mylist)
			print(len(lis))
			writer.writerow(lis)	
			#print(mylist)
					