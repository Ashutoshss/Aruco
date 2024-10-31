################### IMPORT MODULES #######################

import cv2 as cv
import numpy as np
from cv2 import aruco
import os

#######################  code  #######################

#step 1: marke a object of the aruco modules dictionary
aruco_dict=aruco.getPredefinedDictionary(aruco.DICT_4X4_50)

    #step 2: set the marker size in pixels
marker_size=400
num=49  #the number of marker to generate

#check if the directory is present or not
# directory_path="/Markers"
# if not os.path.exists(directory_path):
#     os.makedirs(directory_path)
#     print(f"the directory {directory_path} is created.")
# else:
#     print("directory already exist.")

#step 3:
for ids in range(num):
    marker_image=aruco.generateImageMarker(aruco_dict,ids,marker_size)
    cv.imshow("marker:",marker_image)
    cv.imwrite(f"Markers/marker_{ids}.png",marker_image)