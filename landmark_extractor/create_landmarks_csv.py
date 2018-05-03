import csv
import os 

csv_dir = "landmarks.csv"
dir_path = "/home/saurav/Documents/IIIT-CFW1.0/landmarks/landmarks_new/"

all_data = []
for file in os.listdir(dir_path):
	print(file)
	filepath = dir_path + file
	with open(filepath, 'r+') as f:
		line = f.readlines()
		print(line)
		all_data.append(line)

with open(csv_dir, 'a', newline='\n') as outf:
	writer = csv.writer(outf, delimiter=',', lineterminator='\n')
	writer.writerows(all_data)

