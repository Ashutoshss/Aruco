import cv2
import cv2.aruco as aruco

# Create an empty image for the marker
marker_size = 200
marker_id = 23
border_bits = 1
dictionary = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
markerImage = aruco.drawMarker(dictionary, marker_id, marker_size, borderBits=border_bits)

# Save the marker image
cv2.imwrite("marker23.png", markerImage)
