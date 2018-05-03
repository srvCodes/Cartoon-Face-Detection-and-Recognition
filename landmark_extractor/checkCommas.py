import os
import csv 

csv_dir = "landmarks.csv"

with open(csv_dir, 'r+') as f:
	reader = csv.reader(f,delimiter=',')
	for i, line in enumerate(reader):
		if(len(line)!=31):
			print(line[0])
