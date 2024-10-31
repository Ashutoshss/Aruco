import cv2 as cv
from cv2 import aruco
import numpy as np

# Create ArUco marker dictionary and parameters
marker_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
param_marker = aruco.DetectorParameters()

calib_data_path ="../aruco/calib_data/MultiMatrix.npz"    #calib data path
calib_data=np.load(calib_data_path)         #to load the image to the script
#print(calib_data.file)      #to access the file names

camera_matrix =calib_data["camMatrix"]
dist_coeffs=calib_data["distCoef"]
r_vector=calib_data["rVector"]
t_vector=calib_data["tVector"]
# Open a video capture device
capture = cv.VideoCapture(0)

# Create a 3D points array for the marker corners
marker_length = 0.1  # Length of the marker's side in meters
obj_points = np.array([[-marker_length/2, marker_length/2, 0],
                       [marker_length/2, marker_length/2, 0],
                       [marker_length/2, -marker_length/2, 0],
                       [-marker_length/2, -marker_length/2, 0]], dtype=np.float32)

while True:
    isTrue, frame = capture.read()
    gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    
    # Detect markers and estimate their pose
    marker_corners, marker_IDs, _ = aruco.detectMarkers(gray_frame, marker_dict, parameters=param_marker)
    
    if marker_corners:
        for i in range(len(marker_IDs)):
            # Estimate the pose of each detected marker
            rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(marker_corners[i], marker_length, camera_matrix, dist_coeffs)
            
            # Draw the axis on the marker
            cv.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvecs[i], tvecs[i], 0.1)
            
            # Draw the ID on the marker
            aruco.drawDetectedMarkers(frame, marker_corners)
    
    # Display the frame
    cv.imshow("camera1", frame)
    
    if cv.waitKey(20) & 0xFF == ord('d'):
        break

# Release the camera capture and close OpenCV windows
capture.release()
cv.destroyAllWindows()
