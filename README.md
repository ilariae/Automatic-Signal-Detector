# Automatic-Signal-Detector
Automatically detect hand gestures using the laptop camera and opencv

Project originally done on google colab, just putting the code also on here. 

## University Computer Vision project 

Below the description of the different tasks implemented.

**Task 1**: In this first task, I identified the face using a Haarcascade over a converted grayscale image. I used a gray image as they have less information to process, improving speed and efficiency compared to detection over a colored image.
The region of interest is then identified and drawn with the coordinates identified in the detect function. It is important to note that the image is defined from the top left at 0 ,0 to the bottom right.

**Task 3**: For the third task I am going to use facedetect once to get the face and compute the region of interest. I am then switching to camshift to calculate the hystogram of the face. The algoritmh then calculates on the window where its the most probable to get a face and searches in the region of interest where is the face that has the same distribution.

**Task 4**: In the fourth task the goal is to remove the face so we can detect the hands. I am goin to remove the face from the probability map so the algorithm will look at the picture to find where there is a similar distribution of color as the face and it will find the hands.

**Task 5**: In task 5 the scope is to detect and store the hand as an image in two different sizes: 16x16 and 224x224.
We start by asking how many pictures you want to take and how many seconds between each picture.
We continue by actually taking the pictures, cropping them so it takes just the hand and saving them.

**Task 6**: In task 6 we create our dataset, the letters chosen were M, N and W.

**Task 7**: In task 7 task we build our MLP.

Comparison: After training the 3 models we are going to see how each models performs on each dataset.
Reminder:
Dataset 1: 3 letters with equal number of pictures and a lot of variability
Dataset 2: 3 letters with unbalanced number of pictures (50 - 100 -150) and a lot of variability
Dataset 3: 3 letters with equal number of pictures and one of them with no variability (N)

---
-- Model 1 --

dataset1.txt
```sh
210 train samples - 90 test samples
Validation loss: 1.4552514553070068
Validation accuracy: 0.6555555462837219
```

dataset2.txt
```sh
244 train samples - 106 test samples
Validation loss: 0.9062689542770386
Validation accuracy: 0.8301886916160583
```

dataset3.txt
```sh
210 train samples - 90 test samples
Validation loss: 0.669061005115509
Validation accuracy: 0.8444444537162781
```

Model 1 performs best with dataset 2 and 3, this is probably due to the unbalanced number of pictures in dataset 2 and lack of variability in dataset 3.

---
-- Model 2 --

dataset1.txt
```sh
210 train samples - 90 test samples
Validation loss: 1.7094557285308838
Validation accuracy: 0.7666666507720947
```

dataset2.txt
```sh
244 train samples - 106 test samples
Validation loss: 0.9224103689193726
Validation accuracy: 0.8396226167678833
```

dataset3.txt
```sh
210 train samples - 90 test samples
Validation loss: 1.2521220445632935
Validation accuracy: 0.7555555701255798
```

Model 2 performs best with its own dataset. To make it simple, it's probably easier for the model to guess the right letter when one letter has such an high number of pictures compared to the others. There's a lot more probability that it's going to be the letter with the highest number of pictures.

---
-- Model 3 --

dataset1.txt
```sh
210 train samples - 90 test samples
Validation loss: 1.2043957710266113
Validation accuracy: 0.7888888716697693
```

dataset2.txt
```sh
244 train samples - 106 test samples
Validation loss: 1.269263744354248
Validation accuracy: 0.7641509175300598
```

dataset3.txt
```sh
210 train samples - 90 test samples
Validation loss: 1.8944649696350098
Validation accuracy: 0.699999988079071
```

Model 3 performs best with datase 1 and 2.

Task 8: Test phase: for the test phase I am going to use model 1. I show the hand to the camera doing some of the letters with which I trained the model.
The program finds my hand, generates a gray scale image of the probability of your hand, resize the image to (1,256).
Pass this image to the loaded model and predict.
I show the prediction with a text in the video.
