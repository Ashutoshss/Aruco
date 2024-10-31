#######################  Imported Modules  #######################
import numpy as np
import cv2 as cv
from cv2 import aruco

#######################     code    #######################

# creating the object of the dictionary 
aruco_dict =aruco.getPredefinedDictionary(aruco.DICT_4X4_50)

#Parameter
para_marker=aruco.DetectorParameters()

capture=cv.VideoCapture(0)

while True:
    isTrue,frame=capture.read()

    gray_frame=cv.cvtColor(frame,cv.COLOR_BGR2GRAY)     #convert the bgr to gray
    
    ########################    function to detect the marker   #######################
    marker_corners, marker_IDs, reject = aruco.detectMarkers(gray_frame,aruco_dict,parameters=para_marker)
    # print(marker_corners)
    # print(marker_IDs)   #to print the marker ids in the terminal
    if marker_corners:      #because if no marker detected empty value can not be itrated   
        for ids,corners in zip(marker_IDs,marker_corners):
            #draw a border
            cv.polylines(frame,[corners.astype(np.int32)],True,(0,255,0),2,cv.LINE_AA)  #def polylines(img: MatLike,pts: Sequence[MatLike],isClosed: bool,color: Scalar,thickness: int = ...,lineType: int = ...,shift: int = ...)
            corners=corners.reshape(4,2)
            corners=corners.astype(int)
            top_right=corners[0].ravel()     #indexing the origin of the top index of the shape
            
            x=corners[0][0]
            y=corners[0][1]
            x2=corners[1][0]
            y2=corners[1][1]
            width=((x2-x)**2+(y2-y)**2)**(0.5)
            
            width=int(width/10)
            print(width)
            cv.rectangle(frame, (top_right[0] - width, top_right[1] - width),(top_right[0] + width, top_right[1] + width), (0, 255, 0), 2)
            cv.putText(frame,f"ids:{ids}",top_right,cv.FONT_HERSHEY_COMPLEX,1.5,(0,0,255),2,cv.LINE_AA)    #def putText(img: MatLike,text: str,org: Point,fontFace: int,fontScale: float,color: Scalar,thickness: int = ...,lineType: int = ...,bottomLeftOrigin: bool = ...) -> MatLike: ...

    cv.imshow("camera",frame)

    if cv.waitKey(20) & 0xFF ==ord("q"):
        break
capture.release()
cv.destroyAllWindows()