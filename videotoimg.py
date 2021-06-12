import cv2
import os
import numpy as np



cap=cv2.VideoCapture(0,cv2.CAP_DSHOW)
skip=0
face_data=[]
dataset='./data/'

filename=input("Enter the name of the person")

while True:
    face_section=(0,0,0,0)
    ret,test_img=cap.read()# captures frame and returns boolean value and captured image

    face_haar_cascade=cv2.CascadeClassifier('HaarCascade/haarcascade_frontalface_default.xml')   #Load haar classifier
    faces_detected=face_haar_cascade.detectMultiScale(test_img,scaleFactor=1.32,minNeighbors=5)  #detectMultiScale returns rectangles
    
    # fetting the largest face
    faces_detected=sorted(faces_detected,key= lambda f:f[2]*f[3],reverse=True)
    
    for (x,y,w,h) in faces_detected:
      cv2.rectangle(test_img,(x,y),(x+w,y+h),(255,255,0),thickness=2)
      # croppint only faces
      padding=10
      face_section=test_img[y-padding:y+h+padding,x-padding:x+padding+w]
      face_section=cv2.resize(face_section,(100,100))
     
      skip+=1
      if(skip%5==0):
          face_data.append(face_section)
    

    cv2.imshow('face detection',test_img)
    cv2.imshow('face_section',face_section)

    if (cv2.waitKey(10) & 0xFF) == ord('q'):    #wait until 'q' key is pressed
        break


face_data=np.asarray(face_data)
face_data=face_data.reshape((face_data.shape[0],-1))
f_name=dataset+filename+'.npy'
if os.path.exists(f_name):
    old = np.load(f_name)
    data = np.vstack([old,face_data])
    print("file exists")
    np.save(dataset+filename+'.npy',data)
    old = np.load(f_name)
    print(old.shape)
else:
    np.save(dataset+filename+'.npy',face_data)

print("Data successfully saved at :",dataset+filename+'.npy' )

cap.release()
cv2.destroyAllWindows()
