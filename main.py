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

  
img_path = "/content/gdrive/MyDrive/Computer Vision Ilaria/images/data1/"

def resize_images(images, letter): 
  sizes = [16, 224]
  for id,image in enumerate(images):
    for size in sizes:
      file_name = letter.upper()+"_" + str(id) +"_"+  str(size) #give the pic a name
      print(file_name) #print the name
      try:
        resized_img = cv2.resize(image, (size, size), interpolation = cv2.INTER_AREA) #resize image
      except:
        break
      cv2_imshow(resized_img) #print resized pic
      cv2.imwrite(img_path + file_name + ".jpg", resized_img) #save image to folder
      
     
#calling the function, change the letter when needed
resize_images(saved_images, 'N')

# ----- Task 6 -----

# changes the shape of the array of the 16x16 images and appends it into the text file dataset.txt
def Resize_and_Write(id, letter, dataset):
  #given an id and letter, reads the image and converts it into a list
  img = cv2.imread(img_path +letter+"_" + str(id) + "_16.jpg") #
  res = cv2.resize(img, dsize=(1,256), interpolation=cv2.INTER_CUBIC) # 16x16 to (1,256) 1row | 256 columns
  flat_res = [str(a[0][0]) for a in res] # 1 with 256 elem  
  is_empty = False
  try:
    with  open("/content/gdrive/MyDrive/Computer Vision Ilaria/" + dataset, "r") as fileobj: # open the dataset in read mode
      if len(fileobj.readlines())==0: # checks if the dataset is empty
        is_empty = True
  except:
    is_empty = True # if the dataset does not exist, set the empty variable to true
  with open("/content/gdrive/MyDrive/Computer Vision Ilaria/" + dataset, "a+") as fileobj: # open the dataset in append mode
    L_list=[letter, *flat_res] # create a list with the letter as first element, followed by all the values of the pixels
    str_list = str(L_list)
    str_list = str_list.replace('\'', '')[1:-1] # remove the quotes and the brackets
    if is_empty:
      fileobj.write(str_list)#Creates file if file doesnt exist/also appends 
    else:
      fileobj.write("\n" + str_list)#Creates file if file doesnt exist/also appends

letters = ["M", "N", "W"]
dataset_1 = "dataset1.txt"
dataset_2 = "dataset2.txt"
dataset_3 = "dataset3.txt"

#change when needed
n_images = 100

#Below we just change the name of the dataset as needed.
for i in range(n_images):
  Resize_and_Write(i, "N", dataset_3)

#At the end we shuffle the lines of the dataset.
lines = open('/content/gdrive/MyDrive/Computer Vision Ilaria/dataset2.txt').readlines()
random.shuffle(lines)
open('/content/gdrive/MyDrive/Computer Vision Ilaria/dataset2.txt', 'w').writelines(lines)

# ----- Task 7 -----

dataset_file_path = "/content/gdrive/MyDrive/Computer Vision Ilaria/dataset1.txt"

#The following function takes the dataset file and returns two arrays: samples and letters.
def load_dataset(dataset_file_path):
    a = np.loadtxt(dataset_file_path, delimiter=',', converters={ 0 : lambda ch : ord(ch)-ord('A') })
    samples, letters = a[:,1:], a[:,0] # samples takes all the rows and all the columns except the first column
    # letters: keep only the first column
    return samples, letters

#here we split the dataset for training and validation
train_ratio = 0.7
samples, letters = load_dataset(dataset_file_path)
n_train_samples = int(len(samples) * train_ratio)
x_train, y_train = samples[:n_train_samples], letters[:n_train_samples] # keeps only the first 70% of rows
x_val, y_val = samples[n_train_samples:], letters[n_train_samples:] # keeps only the last 30%
print(x_train.shape)

num_classes = 26 # number of letters
epochs = 100 # number of training s


#here we split the dataset for training and validation
train_ratio = 0.7
samples, letters = load_dataset(dataset_file_path)
n_train_samples = int(len(samples) * train_ratio)
x_train, y_train = samples[:n_train_samples], letters[:n_train_samples] # keeps only the first 70% of rows
x_val, y_val = samples[n_train_samples:], letters[n_train_samples:] # keeps only the last 30%

x_train = x_train.astype('float32')
x_val = x_val.astype('float32')
x_train /= 255 # x_train=x_train/255    normalize the numbers in the interval [0,1]
x_val /= 255
print(x_train.shape[0], 'train samples') # print the number of images
print(x_val.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = tf.keras.utils.to_categorical(y_train, num_classes) # converts the prediction letter into a number encoding for that letter
y_val = tf.keras.utils.to_categorical(y_val, num_classes)

model = Sequential() 
model.add(Dense(100, activation='relu', input_shape=(256,))) # add a relu layer
model.add(Dense(100, activation='relu')) # add anothe relu
model.add(Dropout(0.4)) # add a regularizator
model.add(Dense(100, activation='relu')) # add anothe relu
model.add(Dense(100, activation='relu')) # add anothe relu 
model.add(Dropout(0.4)) # add a regularizator
model.add(Dense(num_classes, activation='softmax'))  # add a final softmax layer

model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

history = model.fit(x_train, y_train,   
                    epochs=epochs,
                    verbose=1, # if verbose=1 print data for each epoch
                    validation_data=(x_val, y_val)) # train the network
score = model.evaluate(x_val, y_val, verbose=0) # evaluate the obtained model  on the validation set.  #model and dataset
print('Validation loss:', score[0]) # print the loss
print('Validation accuracy:', score[1]) # print the accuracy

#After I trained the model, I save the model and weights in two files in the shared folder.
model_json = model.to_json()
with open("/content/gdrive/MyDrive/Computer Vision Ilaria/model1.json", "w") as json_file:
    json_file.write(model_json)
model.save_weights("/content/gdrive/MyDrive/Computer Vision Ilaria/model1_weights.h5")
print("Saved model to disk")

# ----- Comparison -----

def evaluate_model(path, model):
  train_ratio = 0.7
  samples, letters = load_dataset(path) # load the dataset
  n_train_samples = int(len(samples) * train_ratio)
  x_train, y_train = samples[:n_train_samples], letters[:n_train_samples] # keeps only the first 70% of rows
  x_val, y_val = samples[n_train_samples:], letters[n_train_samples:] # keeps only the last 30%
  x_train = x_train.astype('float32')
  x_val = x_val.astype('float32')
  x_train /= 255 # x_train=x_train/255    normalize the numbers in the interval [0,1]
  x_val /= 255
  
  print(x_train.shape[0], 'train samples') # print the number of images
  print(x_val.shape[0], 'test samples')

  # convert class vectors to binary class matrices
  y_train = tf.keras.utils.to_categorical(y_train, num_classes) # converts the prediction letter into a number encoding for that letter
  y_val = tf.keras.utils.to_categorical(y_val, num_classes)

  score = model.evaluate(x_val, y_val, verbose=0) # evaluate the model on the validation set.  #model and dataset
  print('Validation loss:', score[0]) # print the loss
  print('Validation accuracy:', score[1]) # print the accuracy
  
  from keras.models import model_from_json
g_path = "/content/gdrive/MyDrive/Computer Vision Ilaria/"
# load json and create model
json_file = open(g_path+'model3.json', 'r') 
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

loaded_model.summary()
# load weights into new model
loaded_model.load_weights(g_path+"model3_weights.h5")

loaded_model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

dataset_file_paths = [g_path+"dataset1.txt", g_path+"dataset2.txt", g_path+"dataset3.txt"]
for path in dataset_file_paths:
  print(path)
  evaluate_model(path, loaded_model)
  
# ----- Task 8 -----
  
# load json and create model
json_file = open('/content/gdrive/MyDrive/Computer Vision Ilaria/model1.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json) #not used
# load weights into new model
loaded_model.load_weights("/content/gdrive/MyDrive/Computer Vision Ilaria/model1_weights.h5")


prediction = loaded_model.predict(x_train[0:1,:]) # where hand_image is the probability image of your hand of size (1,256)
prediction = prediction.argmax()
predicted_letter = chr(ord('A') + prediction)
print(predicted_letter)

VideoCapture()
eval_js('create()')

CascadeFace = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

isFaceDetect=False
margin = 30
restr = 30
n_images = 0
Time = 1
Picture_count = 1000
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
    #gray_roi = gray[bigrect[0][1]:bigrect[0][3], bigrect[0][0]:bigrect[0][2]] # compute the region of interest
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
  cv2.ellipse(im, track_box, (255, 0, 0), 2) # draw the ellipse
  
  if n_images==Picture_count:
    break
  else:
    time.sleep(Time)
  #we crop the image to have only the hand
  cropped_image = im[int(track_window[1]):int(track_window[1]+track_window[3]), int(track_window[0]):int(track_window[0]+track_window[2])]
  
  #save_image=input("Do you want to save the image Y or N: ")
  resized_img = cv2.resize(cropped_image, (16, 16), interpolation = cv2.INTER_AREA)
  resized_img = cv2.resize(resized_img, dsize=(1,256), interpolation=cv2.INTER_CUBIC)
  resized_img = np.array([str(a[0][0]) for a in resized_img])
  resized_img=resized_img.astype('float32')
  resized_img /= 255.0
  #print(resized_img[np.newaxis, :].shape)
  prediction = loaded_model.predict(resized_img[np.newaxis, :]) # where hand_image is the probability image of your hand of size (1,256)
  prediction = prediction.argmax()
  predicted_letter = chr(ord('A') + prediction)

  font = cv2.FONT_HERSHEY_SIMPLEX
  cv2.putText(im, predicted_letter, (int(track_window[0]), int(track_window[1])), font, 1, (0, 255, 255), 2, cv2.LINE_4)

  eval_js('showimg("{}")'.format(image2byte(im)))
  n_images+=1


