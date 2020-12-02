import cv2
import time
import imutils
cam=cv2.VideoCapture(0)
time.sleep(1)#give the time to initialize or open camera

firstFrame=None
area= 500 # configure movement of object
while True:
    _,img=cam.read()
    text="Normal"
    img=imutils.resize(img, width=500)
    grayImg=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    gaussianImg=cv2.GaussianBlur(grayImg,(21,21),0)#smoothing the img
    if firstFrame is None:
        firstFrame= gaussianImg
        continue
    imgDiff= cv2.absdiff(firstFrame,grayImg)#compare bgrimg to current frame
    threshImg=cv2.threshold(imgDiff,25,255,cv2.THRESH_BINARY)[1]#easily detect the movement
    threshImg= cv2.dilate(threshImg, None,iterations=2)#cover the black holes which is in threshold Image
                                            #format of contours, 
    cnts= cv2.findContours(threshImg.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts= imutils.grab_contours(cnts)#some part of image pixels is mixed then find the nebiourhood img and fix it.
    for c in cnts:
        if cv2.contourArea(c)< area:
            continue
        (x,y,w,h)=cv2.boundingRect(c)#it gets the all value of object like x,y,w,h.
        cv2.rectangle(img,(x, y),(x +w,y + h),(0,255,0),2)
        text="Moving Object Detected"
        print(text)
    cv2.putText(img,text, (10,20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255),2)  
    cv2.imshow("CameraFeed",img)
    cv2.imshow("Thresh",threshImg)
    cv2.imshow("Img Diff",imgDiff)
    key= cv2.waitKey(1) & 0xFF
    if key==ord("q"):
        cam.release()
        cv2.destroyAllWindows()
