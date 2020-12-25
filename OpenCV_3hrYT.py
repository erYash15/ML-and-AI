import cv2
import numpy as np
'''
cap = cv2.VideoCapture(0)
cap.set(3,640) # 3 is used for width
cap.set(4,480) # 4 is used for length
cap.set(10,100) #10 is used for brightness

while True:
    success, img = cap.read()
    cv2.imshow('Video', img)
    if cv2.waitKey(1000) & 0xFF == ord('q'):
        break

'''

'''
img = cv2.imread("pancard.png")
kernel = np.ones((5,5), np.uint8)

imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
imgBlur = cv2.GaussianBlur(imgGray, (7,7), 0)
imgCanny = cv2.Canny(img, 100, 200)
imgDilation = cv2.dilate(imgCanny, kernel, iterations = 1)
imgEroded = cv2.erode(imgDilation, kernel, iterations = 1)

#cv2.imshow('Gray Image', imgGray)
#cv2.imshow('Blur Image', imgBlur)
cv2.imshow('Canny Image', imgCanny)
cv2.imshow("Eroded Image", imgEroded)
cv2.waitKey(0)
'''

'''
img = cv2.imread('pancard.png')
print(img.shape)
img_resize = cv2.resize(img,(1000,500))#width:hight


#crop
img_crop = img_resize[0:200,200:500]#hight:width

cv2.imshow('resize', img_resize)
cv2.imshow('crop', img_crop)
'''

'''
img = np.zeros((512,512,3), np.uint8)
#img[200:300,100:500] = 255,0,0

cv2.line(img,(0,0),(300,300), (0,255,0), 10) # strting cordinate, ending cordinate, color in RGB, thinkness
cv2.rectangle(img,(0,0),(250,350),(0,0,100),cv2.FILLED) #at place of filled number can be specified.
cv2.circle(img,(400,50),30,(255,255,0),5) # center, radius

cv2.putText(img," OPENCV ", (300,200), cv2.FONT_HERSHEY_COMPLEX,1,(0,150,0),1) #scale or size,color,thickness

cv2.imshow('image',img)

cv2.waitKey(0)
'''
'''
img = cv2.imread("cards.jpg")
#warp
width,height = 250,350
pts1 = np.float32([[107,218],[287,185],[154,483],[352,442]])
print("pts1",pts1)
pts2 = np.float32([[0,0],[width,0],[0,height],[width,height]])
print("pts2",pts2)
matrix = cv2.getPerspectiveTransform(pts1,pts2)
print("matrix", matrix)
imgOutput = cv2.warpPerspective(img,matrix,(width,length))
cv2.imshow('input',img)
cv2.imshow('output',imgOutput)
cv2.waitKey(10000)
'''

'''
img1 = cv2.imread('pancard.png')
img2 = cv2.imread('cards.jpg')

imgHor = np.hstack((img2,img2))
imgver = np.vstack((img2,img2))

cv2.imshow('hortzontal', img1_gray)
cv2.imshow('Vertical', img1_clr)
cv2.waitKey(0)
'''

'''
def stackImages(scale,imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range ( 0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape [:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y]= cv2.cvtColor( imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank]*rows
        hor_con = [imageBlank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None,scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor= np.hstack(imgArray)
        ver = hor
    return ver
 
img = cv2.imread('cards.jpg')
imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
 
imgStack = stackImages(0.5,([img,imgGray,img],[img,img,img]))
 
# imgHor = np.hstack((img,img))
# imgVer = np.vstack((img,img))
#
# cv2.imshow("Horizontal",imgHor)
# cv2.imshow("Vertical",imgVer)
cv2.imshow("ImageStack",imgStack)
 
cv2.waitKey(0)
'''

'''
def empty(a):
    pass

cv2.namedWindow("TrackBars")
cv2.resizeWindow("TrackBars",640,240)
cv2.createTrackbar("Hue Min", "TrackBars", 38, 179, empty)
cv2.createTrackbar("Hue Max", "TrackBars", 144, 179, empty)
cv2.createTrackbar("Sat Min", "TrackBars", 80, 255, empty)
cv2.createTrackbar("Sat Max", "TrackBars", 220, 255, empty)
cv2.createTrackbar("Val Min", "TrackBars", 127, 255, empty)
cv2.createTrackbar("Val Max", "TrackBars", 255, 255, empty)
cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hmin = cv2.getTrackbarPos("Hue Min", "TrackBars")
    hmax = cv2.getTrackbarPos("Hue Max", "TrackBars")
    smin = cv2.getTrackbarPos("Sat Min", "TrackBars")
    smax = cv2.getTrackbarPos("Sat Max", "TrackBars")
    vmin = cv2.getTrackbarPos("Val Min", "TrackBars")
    vmax = cv2.getTrackbarPos("Val Max", "TrackBars")

    print(hmin,hmax,smin,smax,vmin,vmax)
    lower = np.array([hmin,smin,vmin])
    upper = np.array([hmax,smax,vmax])
    mask = cv2.inRange(imgHSV,lower,upper)
    imgResult = cv2.bitwise_and(img,img,mask=mask)

    cv2.imshow('img',img)
    cv2.imshow("imghsv",imgHSV)
    #cv2.imshow("img",img)
    cv2.imshow("imgmask",mask)
    cv2.imshow("result",imgResult)
    cv2.waitKey(1)
'''


'''
def getContours(img):
    contours, hierarchy = cv2.findContours(img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area>500:
            cv2.drawContours(imgContour, cnt, -1,(255,0,0), 3 )
            peri = cv2.arcLength(cnt,True)
            print (peri)
            approx = cv2.approxPolyDP(cnt,0.02*peri,True) #true for closed
            print(len(approx))
            objCor = len(approx)
            x, y, w, h = cv2.boundingRect(approx)
            
            if objCor == 3: 
                objectType = "Tri"
            else:
                objectType = "None"

            cv2.rectangle(imgContour, (x,y), (x+w,y+h), (0,255,0),2)
            cv2.putText(imgContour,objectType,
                        (x+(w//2)+10,y+(h//2)-10),cv2.FONT_HERSHEY_SIMPLEX,0.5,
                        (0,0,0),2)

path = "shapes.png"

img = cv2.imread(path)
imgContour = img.copy()
imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
imgBlur = cv2.GaussianBlur(imgGray,(7,7),1)
imgCanny = cv2.Canny(imgBlur,50,50)

getContours(imgCanny)


#cv2.imshow('img',img)
#cv2.imshow('imgGray',imgGray)
#cv2.imshow('imgBlur',imgBlur)
cv2.imshow('imgCanny',imgCanny)
cv2.imshow('imgContour', imgContour)

cv2.waitKey(0)
'''

'''
faceCascade = cv2.CascadeClassifier("haarcascades/haarcascade_frontalface_default.xml")
cap = cv2.VideoCapture(0)
while True:
    success, img = cap.read()
    imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(imgGray,1.1,4)
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    cv2.imshow('Video', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
''' 




'''
cap = cv2.VideoCapture(0)

mycolors = [[38,80,127,144,220,255]]

myColorValues = [(255,0,0)]

myPoints = [] #x,y,color_index

def getContours(img):
    contours, hierarchy = cv2.findContours(img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    x,y,w,h =0,0,0,0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area>500:
            #cv2.drawContours(imgResult, cnt, -1,(255,0,0), 3 )
            peri = cv2.arcLength(cnt,True)
            approx = cv2.approxPolyDP(cnt,0.02*peri,True) #true for closed
            x, y, w, h = cv2.boundingRect(approx)
    return x+w//2,y

def drawoncanvas():
    for points in myPoints:
        cv2.circle(imgResult, (points[0],points[1]), 1, myColorValues[points[2]], cv2.FILLED)


def findColor(img):
    newPoints = []
    imgHSV = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    lower = np.array(mycolors[0][0:3])
    upper = np.array(mycolors[0][3:6])
    mask = cv2.inRange(imgHSV,lower,upper)
    x,y = getContours(mask)
    cv2.circle(imgResult,(x,y),1,myColorValues[0],cv2.FILLED)
    #cv2.imshow('img',mask)
    if x!=0 and y!=0:
        newPoints.append([x,y,0])
    return newPoints


while True:
    success,img = cap.read()
    imgResult = img.copy()
    newPoints = findColor(img)
    if len(newPoints) != 0:
        myPoints.extend(newPoints)
    if len(myPoints)!=0:
        drawoncanvas()
    cv2.imshow('video',img)
    cv2.imshow('imgResult',imgResult)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
'''


'''
frameWidth = 640
frameHeight = 480
cap = cv2.VideoCapture("pancard.mp4")
cap.set(3, frameWidth)
cap.set(4, frameHeight)

widthIMG = 640
heightIMG = 480

def getContours(img):
    biggest = np.array([])
    maxArea = 0
    contours, hierarchy = cv2.findContours(img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    x,y,w,h =0,0,0,0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area>2500:
            cv2.drawContours(imgResult, cnt, -1,(255,0,0), 3 )
            peri = cv2.arcLength(cnt,True)
            approx = cv2.approxPolyDP(cnt,0.02*peri,True) #true for closed
            if area > maxArea and len(approx) == 4 :
                biggest = approx
                maxArea = area
    return biggest

def preProcessing(img):
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray,(5,5),1)
    imgCanny = cv2.Canny(imgBlur,200,200)
    kernel = np.ones((5,5))
    imgDial = cv2.dilate(imgCanny,kernel,iterations=2)
    imgThres = cv2.erode(imgDial, kernel,iterations=1)
    return imgThres

while True:
    success,img = cap.read()
    img = cv2.resize(img, (widthIMG,heightIMG))
    imgCountour = img.copy()
    cv2.imshow('thres', preProcessing(img))
    cv2.imshow("imgcount", imgCountour)
    cv2.imshow("Result", img)
    if cv2.waitKey(100) & 0xFF == ord('q'):
        break

'''