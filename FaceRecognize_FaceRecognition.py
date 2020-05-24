
# coding: utf-8

# In[1]:


import cv2, sys, numpy, os


# In[2]:


size=4
haar_file= 'F:/2_Naveen_Python/5_JupyterNotebook/Face/haarcascade_frontalface_default.xml'
datasets= 'F:/2_Naveen_Python/5_JupyterNotebook/datasets'


# Part-1:  Create FisherRecognizer

# In[3]:


print('Recognizing Face...Please be in sufficient light conditions')
#Create a list of images and list of corresponding names

(images, labels, names, id)= ([],[],{},0)

for (subdirs, dirs, files) in os.walk(datasets): 
    for subdir in dirs:
        names[id]= subdir
        subjectpath= os.path.join(datasets, subdir)
        for filename in os.listdir(subjectpath):
            path= subjectpath+ '/' +filename
            label= id
            images.append(cv2.imread(path,0))
            labels.append(int(label))
        id +=1
(width, height)= (130,100)

#Create a Numpy array from the two list above:
(images, labels)= [numpy.array(lis) for lis in [images, labels]]

#OpenCV trains a model from the images...
model= cv2.face.LBPHFaceRecognizer_create()
model.train(images, labels)


# Part-2: Use FaceRecognizer on camera stream

# In[4]:


face_cascade= cv2.CascadeClassifier(haar_file)
webcam= cv2.VideoCapture(0)

while True:
    (_, im)= webcam.read()                             #Reading the frame from the video
    gray= cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)         #Converting colour image to gray scale image
    faces= face_cascade.detectMultiScale(gray, 1.3,5)  #Detecting faces
    for (x,y,w,h) in faces:
        cv2.rectangle(im, (x,y), (x+w, y+h), (255,0,0),2) #To draw rectangel on an image.
        face= gray[y:y+h, x:x+w]
        face_resize= cv2.resize(face, (width, height))    #Resizing the face as per custom width and height
        #Try to recognize the face...
        prediction= model.predict(face_resize)
        cv2.rectangle(im, (x,y), (x+w, y+h), (0,255,0), 3)#To draw rectangle on an image.
        
        if prediction[1]<500:
            cv2.putText(im, '%s - %.0f' %(names[prediction[0]], prediction[1]), (x-10, y-10), cv2.FONT_HERSHEY_PLAIN, 3, (0,255,0))
        else:
            cv2.putText(im, 'Not_Recognized', (x10,y-10), cv2.FONT_HERSHEY_PLAIN, 3, (0,255,0))
            
    cv2.imshow('OpenCV', im)
    
    key= cv2.waitKey(10)
    if key== 27:
        break

