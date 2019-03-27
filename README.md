# Cartoon-Face-Detection-and-Recognition [![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/)

This repo contains the codes necessary to reproduce the results of the paper [Towards Improved Face Detection and Recognition Systems](https://arxiv.org/abs/1804.01753).

The project maintains the following directories:

- [preprocessing](https://github.com/Saurav0074/Cartoon-Face-Detection-and-Recognition/tree/master/preprocessing) contains the files for parsing the XML files to filter the images based on the attributes required by the sub-tasks of the project (i.e., class-wise, gender-wise, facial posture wise manners).

- [landmark_extractor](https://github.com/Saurav0074/Cartoon-Face-Detection-and-Recognition/tree/master/landmark_extractor) contains the code for managing the 15 facial landmarks extraction, arranging these to the format compatible of being merged with the Kaggle instances and training the 5 layer LeNet architecture for landmark extraction. The output csv is further used in the character and gender recognition of the cartoon faces.

- [datasets](https://github.com/Saurav0074/Cartoon-Face-Detection-and-Recognition/tree/master/datasets) hosts two files:
 
  - [train_own.csv](https://github.com/Saurav0074/Cartoon-Face-Detection-and-Recognition/blob/master/datasets/train_own.csv) is the output of running [append_pixels_new.py](https://github.com/Saurav0074/Cartoon-Face-Detection-and-Recognition/blob/master/landmark_extractor/append_pixels_new.py) on [onlyLandmarks.csv](https://github.com/Saurav0074/Cartoon-Face-Detection-and-Recognition/blob/master/datasets/onlyLandmarks.csv), that contains comma separated landmark coordinates manually extracted using [LandmarkManuallyGetter.jar](https://github.com/Saurav0074/Cartoon-Face-Detection-and-Recognition/blob/master/LandmarkManuallyGetter.jar).
  
- [face_detection](https://github.com/Saurav0074/Cartoon-Face-Detection-and-Recognition/tree/master/face_detection) contains the code for running the MTCNN, Haar and HOG based models.

- [face_recognition](https://github.com/Saurav0074/Cartoon-Face-Detection-and-Recognition/tree/master/face_recognition) contains the code for the character recognition of the cartoons based on the Inception v3+SVM and the proposed HCNN model.

- [gender_recognition](https://github.com/Saurav0074/Cartoon-Face-Detection-and-Recognition/tree/master/gender_recognition) contains the code for the character recognition of the cartoons based on the Inception v3+SVM and the proposed HCNN model.

- [outputs](https://github.com/Saurav0074/Cartoon-Face-Detection-and-Recognition/tree/master/outputs) contains the accuracy and top-5 error rate graphs for the character recognition problem, the model architectures used as well as the results of the face detection models.

## Outputs

1. MTCNN face detection

![MTCNNfacedetect](https://github.com/Saurav0074/Cartoon-Face-Detection-and-Recognition/blob/master/outputs/faceDetectionbyMTCNN.png)

2. OpenCV face detection 

![Opencv](https://github.com/Saurav0074/Cartoon-Face-Detection-and-Recognition/blob/master/outputs/faceRecognitionByOpenCv.png)

3. dlib face detection 

![dlib](https://github.com/Saurav0074/Cartoon-Face-Detection-and-Recognition/blob/master/outputs/faceRecognitionDlib.png)

4. Erroneously predicted landmark points

![alt text](https://github.com/Saurav0074/Cartoon-Face-Detection-and-Recognition/blob/master/outputs/Figure_3.png)

## References

S. Jha, N. Agarwal, and S. Agarwal, “Towards Improved Cartoon Face Detection and Recognition Systems,” 2018, arXiv:1804.01753v1.
