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

#VideoCapture creates a real time video stream 

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
'''from google.colab import drive
drive.mount('/content/gdrive')'''

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
    
# ----- TASK 1 -----
#detect face and eyes using the Haar Feature-based Cascade Classifiers

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
  
# ------ TASK 2 ------
#function to compute the bigger rectangle we are going to use as region of interest
def compute_big_rect(rect): #[x1,y1,x2,y2]

  rect[0]=max(0,rect[0]-margin)
  rect[1]=max(0,rect[1]-margin)
  rect[2]=min(rect[2]+margin,image_max_width)
  rect[3]=min(rect[3]+margin,image_max_height)
  
  return rect

#reduce the computation time of Facedetect

VideoCapture()
eval_js('create()')

#application of the Haarcascade for object detection faces
CascadeFace = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

margin = 60
while True:
  bigrect=[]
  e1 = cv2.getTickCount()
  byte = eval_js('capture()')
  im = byte2image(byte)
  gray = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY) # convert the image into GRAY
  face = detect(gray, cascade = CascadeFace) # face contains the rectangle
  #draw_rects(im, face, (0, 0, 0)) # draw the rectangle
  e2 = cv2.getTickCount()
  isDetection=False

  if len(face)==0:
    isFaceDetect=False
  else:
    isFaceDetect=True

  if len(bigrect)==0 and isFaceDetect:
    bigrect.append(compute_big_rect(face[0]))
    isDetection=True

  #this should be the new main cycle instead of the for cycle
  while isDetection:
    e3 =cv2.getTickCount()
    byte = eval_js('capture()')
    im = byte2image(byte)
    gray = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
    
    roi=gray[bigrect[0][1]:bigrect[0][3], bigrect[0][0]:bigrect[0][2]]
    roi_color=im[bigrect[0][1]:bigrect[0][3], bigrect[0][0]:bigrect[0][2]]
    newface=detect(roi, cascade = CascadeFace) 
    
    draw_rects(im,bigrect, (0,0,0))
    draw_rects(roi_color,newface, (153, 255, 255))
    e4 = cv2.getTickCount()

    if len(newface)>0:
      eval_js('showimg("{}")'.format(image2byte(im)))
    else:
      isDetection=False
      bigrect=[]
      isFace=False
      
# ----- Task 3 -----

def compute_big_rect2(rect): #[x1,y1,x2,y2]

  x1=max(0,rect[0]-margin)
  y1=max(0,rect[1]-margin)
  x2=min(rect[2]+margin,image_max_width)
  y2=min(rect[3]+margin,image_max_height)
  
  return [x1,y1,x2,y2]

VideoCapture()
eval_js('create()')

CascadeFace = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

isFaceDetect=False
margin = 30

while not isFaceDetect:
  bigrect=[]
  byte = eval_js('capture()')
  im = byte2image(byte)
  gray = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY) # convert the image into GRAY
  face = detect(gray, cascade = CascadeFace) # face contains the rectangle
  hsv =  cv2.cvtColor(im, cv2.COLOR_RGB2HSV) # convert image to HSV
  
  #control if a face has been detected
  if len(face)==0: 
    isFaceDetect=False
  else:
    isFaceDetect=True
    
  if len(bigrect)==0 and isFaceDetect: # if bigrect is empty and a face is found
    hsv_roi=hsv[face[0][1]:face[0][3], face[0][0]:face[0][2]]
    bigrect.append(compute_big_rect2(face[0])) # compute bigrect and add it to the list
  
    isDetection=True # set a flag to start camshift
    gray_roi = gray[bigrect[0][1]:bigrect[0][3], bigrect[0][0]:bigrect[0][2]] # compute the region of interest
    hist=cv2.calcHist([hsv_roi], [0], None, [180], [0,180]) # compute the histogram
    plt.hist(gray_roi.ravel(),256,[0,256]) # show the histogram
    plt.show()
  
while isDetection:
  byte = eval_js('capture()')
  im = byte2image(byte)
  gray = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY) # convert the image into GRAY
  hsv =  cv2.cvtColor(im, cv2.COLOR_RGB2HSV) #convert image to color

  track_window = (bigrect[0][0], bigrect[0][1], bigrect[0][2], bigrect[0][3]) # compute the region of interest (bigrect)
  prob = cv2.calcBackProject([hsv], [0], hist, [0, 256], 1) # compute the probability matrix 
  term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )  # sets some criteria for camshift
  track_box, track_window = cv2.CamShift(prob, track_window, term_crit) # perform camshift over the track_window

  im[:] = prob[...,np.newaxis] # add probability matrix to the image

  draw_rects(im, face, (255, 0, 255))
  draw_rects(im, bigrect, (255, 0, 0), 4) # draw the rectangle

  try:
    cv2.ellipse(im, track_box, (0, 0, 255), 2) # draw the ellipse
  except Exception as e:
    print(e)
    print(track_box)

  eval_js('showimg("{}")'.format(image2byte(im)))
  
# ----- Task 4 -----

#implement
def restrict(rect): #[x1,y1,x2,y2]

  x1=min(image_max_width,rect[0]+restr)
  y1=min(image_max_height,rect[1]+restr)
  x2=max(rect[2]-restr,x1)
  y2=max(rect[3]-restr,y1)
  return [x1,y1,x2,y2]

VideoCapture()
eval_js('create()')

CascadeFace = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

isFaceDetect=False
margin = 30
restr = 30

while not isFaceDetect:
  bigrect=[] 
  byte = eval_js('capture()')
  im = byte2image(byte)
  gray = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY) # convert the image into GRAY
  hsv =  cv2.cvtColor(im, cv2.COLOR_RGB2HSV) #convert image to color
  face = detect(gray, cascade = CascadeFace) # face contains the rectangle
 
  if len(face)==0:
    isFaceDetect=False
  else:
    isFaceDetect=True
    
  if isFaceDetect: # if a face is found
    restricted_face = restrict(face[0])
    hsv_roi=hsv[restricted_face[1]:restricted_face[3], restricted_face[0]:restricted_face[2]] # !!! better to restrict this face for the histogram
    bigrect.append(compute_big_rect2(face[0])) # compute bigrect and add to the list
  
    isDetection=True # set a flag to start camshift
    #gray_roi = gray[bigrect[0][1]:bigrect[0][3], bigrect[0][0]:bigrect[0][2]] # compute the region of interest
    hist=cv2.calcHist([hsv_roi], [0], None, [180], [0,180]) # compute the histogram
    plt.hist(hsv_roi.ravel(),256,[0,256]) # show the histogram
    plt.show()

    track_window = (bigrect[0][0], bigrect[0][1], bigrect[0][2], bigrect[0][3]) # compute the region of interest (bigrect)

term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 ) # sets some criteria for camshift

while isDetection:
  byte = eval_js('capture()')
  im = byte2image(byte)
  #gray = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY) # convert the image into GRAY
  hsv =  cv2.cvtColor(im, cv2.COLOR_RGB2HSV) #convert image to color
  mask = cv2.inRange(hsv, np.array((0,64,32)), np.array((180,200,200)))

  prob = cv2.calcBackProject([hsv], [0], hist, [0, 180], 1) # compute the probability matrix 

  for i in range(bigrect[0][0], bigrect[0][2]): # iterates between x1, x2
    for j in range(bigrect[0][1], bigrect[0][3]): # iterates between y1, y2
      prob[j,i]=0 # sets probability matrix to 0 on all points inside bigrect

  prob &= mask
  track_box, track_window = cv2.CamShift(prob, track_window, term_crit) # perform camshift over the track_window 
  im[:] = prob[...,np.newaxis] # add probability matrix to the image
  draw_rects(im, [restricted_face], (255, 0, 255)) 
  cv2.ellipse(im, track_box, (255, 0, 0), 2) # draw the ellipse  

  eval_js('showimg("{}")'.format(image2byte(im)))

# ----- Task 5 -----

Picture_count=int(input("How many pictures do you want to take? "))# the number of picture to be taken
Time=int(input("How many seconds between each photo? "))
n_images=0 # counter of saved images
saved_images =[]

VideoCapture()
eval_js('create()')

CascadeFace = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

isFaceDetect=False
margin = 30
restr = 30

while not isFaceDetect:
  bigrect=[] 
  byte = eval_js('capture()')
  im = byte2image(byte)
  gray = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY) # convert the image into GRAY
  hsv =  cv2.cvtColor(im, cv2.COLOR_RGB2HSV) #convert image to color
  face = detect(gray, cascade = CascadeFace) # face contains the rectangle
 
  if len(face)==0:
    isFaceDetect=False
  else:
    isFaceDetect=True
    
  if isFaceDetect: # if a face is found
    restricted_face = restrict(face[0])
    hsv_roi=hsv[restricted_face[1]:restricted_face[3], restricted_face[0]:restricted_face[2]] # !!! better to restrict this face for the histogram
    mask_roi = cv2.inRange(hsv_roi, np.array((0., 60., 32.)), np.array((180., 255., 255.)))
    bigrect.append(compute_big_rect2(face[0])) # compute bigrect and add to the list
  
    isDetection=True # set a flag to start camshift
    hist=cv2.calcHist([hsv_roi], [0], mask_roi, [180], [0,180]) # compute the histogram
    hist=cv2.normalize(hist, hist,  alpha=0, beta=255, norm_type=cv2.NORM_MINMAX) # normalize the histogram 
    plt.hist(hsv_roi.ravel(),256,[0,256]) # show the histogram
    plt.show()

    track_window = (bigrect[0][0], bigrect[0][1], bigrect[0][2], bigrect[0][3]) # compute the region of interest (bigrect)

term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 ) # sets some criteria for camshift

while isDetection:
  byte = eval_js('capture()')
  im = byte2image(byte)
  hsv =  cv2.cvtColor(im, cv2.COLOR_RGB2HSV) #convert image to color
  mask = cv2.inRange(hsv, np.array((0,64,32)), np.array((180,200,200)))
  raw_im = cv2.cvtColor(gray,cv2.COLOR_GRAY2RGB)
  prob = cv2.calcBackProject([hsv], [0], hist, [0, 180], 1) # compute the probability matrix 
  

  for i in range(bigrect[0][0], bigrect[0][2]): # iterates between x1, x2
    for j in range(bigrect[0][1], bigrect[0][3]): # iterates between y1, y2
      prob[j,i]=0 # sets probability matrix to 0 on all points inside bigrect

  prob &= mask
  track_box, track_window = cv2.CamShift(prob, track_window, term_crit) # perform camshift over the track_window  
  im[:] = prob[...,np.newaxis] # add probability matrix to the image
  draw_rects(im, [restricted_face], (255, 0, 255))

  eval_js('showimg("{}")'.format(image2byte(im)))
  if n_images==Picture_count:
    break
  else:
    time.sleep(Time)
  #we crop the image to have only the hand
  cropped_image = im[int(track_window[1]):int(track_window[1]+track_window[3]), int(track_window[0]):int(track_window[0]+track_window[2])]
  cv2_imshow(cropped_image)  
  save_image=input("Do you want to save the image Y or N: ")
  
  if str.lower(save_image) == 'y':

    saved_images.append(cropped_image)
    print("Picture",n_images+1,"was stored")
    #print(cropped_image)
    n_images+=1
    continue
  else:
    continue

#to see all the saved pictures
for image in saved_images:
  cv2_imshow(image)

# ----- Task 6 -----









