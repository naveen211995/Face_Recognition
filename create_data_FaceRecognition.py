
# coding: utf-8

# In[1]:


import cv2, sys, numpy, os


# In[2]:


haar_file= 'F:/2_Naveen_Python/5_JupyterNotebook/Face/haarcascade_frontalface_default.xml'
datasets= 'F:/2_Naveen_Python/5_JupyterNotebook/datasets'        #All the faces data will be present in this folder
sub_data= 'F:/2_Naveen_Python/5_JupyterNotebook/Bhavani'        #These are the sub-datasets of the above folder, for my face I have used my name.


# In[3]:


path= os.path.join(datasets, sub_data)
if not os.path.isdir(path):  #checking whether such directory exists,
  os.mkdir(path)             # if not then we will such directory
(width, height)= (130,100)   #Defining the size of the image.


# In[4]:


face_cascade= cv2.CascadeClassifier(haar_file)  #Making use of cascade function that is trained from a lot of faces and non-faces
webcam= cv2.VideoCapture(0) #'0' is used for my webcam, If you have any other webcam attached then use '1'


# In[5]:


#The program loops until it has 30 images of the face...
count=1
while count<30:
  (_, im)= webcam.read()                             #Reading the frame from the video
  gray= cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)         #Converting colour image to gray scale image
  faces= face_cascade.detectMultiScale(gray, 1.3, 4) #Detecting faces
  for (x,y,w,h) in faces:
    cv2.rectangle(im, (x,y), (x+w, y+h), (255,0,0),3)#To draw rectangel on an image.
    face= gray[y: y+h, x: x+w]                      
    face_resize= cv2.resize(face, (width, height))   #Resizing the face as per custom width and height
    cv2.imwrite('%s/%s.png' % (path,count), face_resize) #Writing the saved images
  count +=1

  cv2.imshow('OpenCV', im)
  key= cv2.waitKey(10)
  if key==27:
    break

