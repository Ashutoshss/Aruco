import cv2 as cv
import os

CHESS_BOARD_DIM = (9, 6)

n = 0  # image_counter


##########################   checking the image dir exist or not   ################################


# checking if  images dir is exist not, if not then create images directory
image_dir_path = "images"

CHECK_DIR = os.path.isdir(image_dir_path)   #return boolean vlaue

# if directory does not exist create
if not CHECK_DIR:
    os.makedirs(image_dir_path)
    print(f'"{image_dir_path}" Directory is created')
else:
    print(f'"{image_dir_path}" Directory already Exists.')


###################################################################################################


criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

############################  function the detect the checkerboard  ###############################

def detect_checker_board(image, grayImage, criteria, boardDimension):
    ret, corners = cv.findChessboardCorners(grayImage, boardDimension)
    if ret == True:
        corners1 = cv.cornerSubPix(grayImage, corners, (3, 3), (-1, -1), criteria)
        image = cv.drawChessboardCorners(image, boardDimension, corners1, ret)

    return image, ret


###################################################################################################


########################################  video capture part   ####################################
capture = cv.VideoCapture(0)

while True:
    _, frame = capture.read()
    copyFrame = frame.copy()
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    image, board_detected = detect_checker_board(frame, gray, criteria, CHESS_BOARD_DIM)
    # print(ret)
    cv.putText(frame,f"saved_img : {n}",(30, 40),cv.FONT_HERSHEY_PLAIN,1.4,(0, 255, 0),2,cv.LINE_AA)

    
################################################################################################

    cv.imshow("frame", frame)
    cv.imshow("copyFrame", copyFrame)       



    if cv.waitKey(20) & 0xFF== ord("q"):
        break
    if 0xFF== ord("s") and board_detected == True:      #to save the image
        # storing the checker board image
        cv.imwrite(f"{image_dir_path}/image{n}.png", copyFrame)

        print(f"saved image number {n}")
        n += 1  # incrementing the image counter

capture.release()
cv.destroyAllWindows()

print("Total saved Images:", n)