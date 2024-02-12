import csv
import cv2
import numpy as np
from keras.models import model_from_json
from tkinter import *
from datetime import date
def proj2():
    emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

    # load json and create model
    json_file = open('emotion_model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    emotion_model = model_from_json(loaded_model_json)

    # load weights into new model
    emotion_model.load_weights("emotion_model.h5")
    print("Loaded model from disk")

    # start the webcam feed
    cap = cv2.VideoCapture(0)

    
    global c
    while True:
        # Find haar cascade to draw bounding box around face
        ret, frame = cap.read()
        frame = cv2.resize(frame, (1280, 720))
        if not ret:
            break
        face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # detect faces available on camera
        num_faces = face_detector.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

        # take each face available on the camera and Preprocess it
        for (x, y, w, h) in num_faces:
            cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (0, 255, 0), 4)
            roi_gray_frame = gray_frame[y:y + h, x:x + w]
            cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)

            # predict the emotions
            emotion_prediction = emotion_model.predict(cropped_img)
            maxindex = int(np.argmax(emotion_prediction))
            emotiondetect=emotion_dict[maxindex]
            cv2.putText(frame, emotion_dict[maxindex], (x+5, y-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
           
        cv2.imshow('Emotion Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            return emotion_dict[maxindex]

    cap.release()
    cv2.destroyAllWindows()
    
    
    
def proj1():
    
    root=Tk()
    dictiona={14800219059:"Sayan059.csv",14800219036:"Sayan036.csv"}
    admi=[14800219059]
    
    def getvals():
        if(fvalue.get()==1 and fvalue1.get()==0):
            if(idnvalue.get() in dictiona):
                fil=open(dictiona[idnvalue.get()],'a')
                csvwriter=csv.writer(fil)
                tod=str(date.today())
                emotiondetect=proj2()
                p=[tod,emotiondetect]
                csvwriter.writerow(p)
                fil.close()
        elif(fvalue.get()==0 and fvalue1.get()==1 and idnvalue.get() in admi):

            if(vaval.get() in dictiona):
                d={}
                ap=[]
                fil=open(dictiona[vaval.get()],'r')
                csvreader=csv.reader(fil)
                for i in csvreader:
                    if(len(i)>0):
                        ap.append(i[1])
                fil.close()
                for i in ap:
                    if(i in d):
                        d[i]=d[i]+1
                    else:
                        d[i]=1
                if("" in d):
                    del d[""]
                m=0
                j=""
                for i in d:
                    if(d[i]>m):
                        m=d[i]
                        j=i
                print("Employee Job Satisfaction : \n",j)
            else:
                print("Not Found")
        else:
            print("not found")
        
            
            
    root.geometry("644x344")
    Label(root, text="WELCOME TO THE ORGANIZATION", font="comicsansas 13 bold",pady=15).grid(row=0,column=3)
    idn=Label(root, text="ID No.")
    idn.grid(row=1,column=2)
    idnvalue=IntVar()
    fvalue=IntVar()
    fvalue1=IntVar()
    idnentry=Entry(root, textvariable=idnvalue)
    idnentry.grid(row=1,column=3)
    ff=Checkbutton(text="EMPLOYEE",variable=fvalue)
    ff1=Checkbutton(text="ADMIN",variable=fvalue1)
    ff.grid(row=6,column=3)
    ff1.grid(row=7,column=3)  
    va=Label(root, text="Search ID no.")
    va.grid(row=8,column=2)
    vaval=IntVar()
    vaentry=Entry(root,textvariable=vaval)
    vaentry.grid(row=8,column=3)
        
    Button(text="SUBMIT ATTENDANCE",command=getvals).grid(row=9,column=3)
    root.mainloop()
proj1()