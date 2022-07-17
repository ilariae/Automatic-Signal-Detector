from IPython.display import display, Javascript
from google.colab.output import eval_js
from base64 import b64decode, b64encode
from google.colab.patches import cv2_imshow
import numpy as np
from PIL import Image
import time
import random

import io
import cv2 # OpenCV library

import matplotlib.pyplot as plt

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
import tensorflow as tf
from keras.models import model_from_json

# VideoCapture creates a real time video stream 

def VideoCapture():
  js = Javascript('''
    // the function 'create' creates the "box" that contains the videostream
    // async functions return a promise, they make the code look synchronous, but it's asynchronous and non-blocking behind the scenes
    async function create(){ 
      div = document.createElement('div');
      document.body.appendChild(div);

      video = document.createElement('video');
      video.setAttribute('playsinline', '');

      div.appendChild(video);
      // await is used to call functions, the calling code will stop until the promise is resolved or rejected
      stream = await navigator.mediaDevices.getUserMedia({video: {facingMode: "environment"}});
      video.srcObject = stream;

      // await is used to call functions, here we call the video.play() function
      await video.play();

      canvas =  document.createElement('canvas');
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      canvas.getContext('2d').drawImage(video, 0, 0);

      div_out = document.createElement('div');
      document.body.appendChild(div_out);
      img = document.createElement('img');
      div_out.appendChild(img);
    }

    //  
    async function capture(){
        return await new Promise(function(resolve, reject){
            pendingResolve = resolve;
            canvas.getContext('2d').drawImage(video, 0, 0);
            result = canvas.toDataURL('image/jpeg', 0.80);

            pendingResolve(result);
        })
    }

    function showimg(imgb64){
        img.src = "data:image/jpg;base64," + imgb64;
    }

  ''')
  display(js)


VideoCapture()
eval_js('create()')

image_max_width = 640
image_max_height = 480

#access google drive to use files
from google.colab import drive
drive.mount('/content/gdrive')

# Given a variable byte containing the bytes of an image, returns an array representing the image
def byte2image(byte):
  jpeg = b64decode(byte.split(',')[1])
  im = Image.open(io.BytesIO(jpeg))
  return np.array(im)

#Given an array representing an image, returns the byte associated with the image
def image2byte(image):
  image = Image.fromarray(image)
  buffer = io.BytesIO()
  image.save(buffer, 'jpeg')
  buffer.seek(0)
  x = b64encode(buffer.read()).decode('utf-8')
  return x

def detect(img, cascade):
  # rects represent the coordinates of the rectangles containing different objects
  rects = cascade.detectMultiScale(img, scaleFactor=1.3, minNeighbors=4, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)
  if len(rects) == 0: # if no rectangles are found, return the empty list
    return []  
  rects[:,2:] += rects[:,:2] #(x1,y1,x2,y2) we have the top left and bottom right corners of the rectangle

  return rects

#draw the rects on the video
def draw_rects(img, rects, color, size=2):
  for x1, y1, x2, y2 in rects:
    cv2.rectangle(img, (x1, y1), (x2, y2), color, size)
    
#TASK 1

# detect face and eyes using the Haar Feature-based Cascade Classifiers

VideoCapture() #starts the VideoCapture function
eval_js('create()') #utilizes the given JS code

image_max_width = 640
image_max_height = 480

#application of the Haarcascade for object detection faces
CascadeFace = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

while True:
  e1 = cv2.getTickCount()

  byte = eval_js('capture()')
  im = byte2image(byte)  

  gray = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY) # converting the image into GRAY reduces the color channels of the image to grayscale and improves computation speed 
  
  face = detect(gray, cascade = CascadeFace) # # applies the Haar face cascade to detect the face int he gray image 
  draw_rects(im, face, (0, 0, 0)) # calls the draw function to draw a rectangle around the face

  e2 = cv2.getTickCount()
  face_time = e2-e1  # - CONVERT INTO SECONDS
  #print("the number of clock-cycles to detect the face is :", face_time)
  e3 = cv2.getTickCount()

  eval_js('showimg("{}")'.format(image2byte(im)))
  
  #TASK 2
