# %%
import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt

#calibration consists in 3 steps:
# 1. find corners in a dataset of images 
# 2. Use corner points to compute a camera matrix
# 3. Use the camera matrix to undistort images

# %%
# read images and convert them to gray

#Load images from stereo_calibration_images and separate the ones taken from right camera and left camera
def readImgs(path):
    imgs_right = glob.glob(f'{path}/left/*.png')
    imgs_left=glob.glob(f'{path}/right/*.png')
    assert imgs_right
    assert imgs_left

    if (len(imgs_left)==len(imgs_right)):
        print("Correct database")

# %%
# 1. FIND CORNERS
def findCorners():
    nb_vertical= 9
    nb_horizontal= 6
    chessboardSize= (9,6)

    frameSize= (1280,720)

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((nb_horizontal*nb_vertical,3), np.float32)
    objp[:,:2] = np.mgrid[0:nb_vertical,0:nb_horizontal].T.reshape(-1,2)

    #objp = objp*33.6 #multiplying with the actual size of the checker board square MIGHT NEED IT


    # %%
    # Arrays to store object points and image points from all the images.
    objpointsL = [] # 3d point in real world space left
    objpointsR = [] # 3d point in real world space right
    imgpointsL = [] # 2d points in image plane left
    imgpointsR = [] # 2d points in image plane right

    # define a criteria for the termination of iteration for looking for corners
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)


    for imgLeft, imgRight in zip(imgs_left, imgs_right):

        imgL = cv2.imread(imgLeft)
        imgR = cv2.imread(imgRight)
        grayL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
        grayR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)

        # Find the chess board corners for all images
        retL, cornersL = cv2.findChessboardCorners(grayL, chessboardSize, None)
        retR, cornersR = cv2.findChessboardCorners(grayR, chessboardSize, None)

        # If found, add object points, image points (after refining them)
        if retL and retR == True:

            objpointsL.append(objp)
            cornersL = cv2.cornerSubPix(grayL, cornersL, (11,11), (-1,-1),criteria)
            imgpointsL.append(cornersL)

            objpointsR.append(objp)
            cornersR = cv2.cornerSubPix(grayR, cornersR, (11,11), (-1,-1), criteria)
            imgpointsR.append(cornersR)

            # Draw and display the corners
            #this can be commented out if we dont want to see the images (faster)
            
            cv2.drawChessboardCorners(imgL, chessboardSize, cornersL, retL)
            cv2.imshow('img left', imgL)
            cv2.drawChessboardCorners(imgR, chessboardSize, cornersR, retR)
            cv2.imshow('img right', imgR)
            cv2.waitKey(1000)


    cv2.destroyAllWindows()



# %%
# 2. USE CORNER POINTS TO COMPUTE CAMERA MATRIX (CALIBRATION)
def Calibrate():
    #Using the extracted corners we can obtain a camera matrix that contains the information needed to undistort images

    #(success, cameramatrix, distorsion parameters, rotation vectors, translation vector)
    retL, cameraMatrixL, distL, rvecsL, tvecsL = cv2.calibrateCamera(objpointsL, imgpointsL, frameSize, None, None) #none as default
    heightL, widthL = imgL.shape[:2] #used to generate a new camera matrix (optimal)
    newCameraMatrixL, roi_L = cv2.getOptimalNewCameraMatrix(cameraMatrixL, distL, (widthL, heightL), 1, (widthL, heightL))

    retR, cameraMatrixR, distR, rvecsR, tvecsR = cv2.calibrateCamera(objpointsR, imgpointsR, frameSize, None, None)
    heightR, widthR = imgR.shape[:2]
    newCameraMatrixR, roi_R = cv2.getOptimalNewCameraMatrix(cameraMatrixR, distR, (widthR, heightR), 1, (widthR, heightR))

    #most important parameters are the camera matrix and distortion parameters to do the calibration


    # %%
    #Stereo Vision Calibration

    flags = cv2.CALIB_RATIONAL_MODEL # flag needed for the stereoCalibrate function

    # This step is performed to transformation between the two cameras and calculate Essential and Fundamenatl matrix
    retStereo, newCameraMatrixL, distL, newCameraMatrixR, distR, rot, trans, essentialMatrix, fundamentalMatrix = cv2.stereoCalibrate(objpointsL, imgpointsL, imgpointsR, newCameraMatrixL, distL, newCameraMatrixR, distR, frameSize, None, None, None, None, flags) #how we relate one camera to the other
    #return values from stereocalibration- 2 new camera matrixes, distortion parameters...

# %%
# 3. USE CAMERA MATRIX TO UNDISTORT IMAGES (Stereo Rectification)
def Undistort():

    # sereorectify: computes and saves rectification of matrix of the 2 cameras (from stereocalibration)
    rectL, rectR, projMatrixL, projMatrixR, Q, roi_L, roi_R= cv2.stereoRectify(newCameraMatrixL, distL, newCameraMatrixR, distR, frameSize, rot, trans, None, None, None, None) #using default parameters 

    stereoMapL = cv2.initUndistortRectifyMap(newCameraMatrixL, distL, rectL, projMatrixL, frameSize, cv2.CV_32FC1)
    stereoMapR = cv2.initUndistortRectifyMap(newCameraMatrixR, distR, rectR, projMatrixR, frameSize, cv2.CV_32FC1)


# %%

# 4. SAVING CAMERA PARAMETERS

def Save():
    cv_file = cv2.FileStorage('stereoMap.xml', cv2.FILE_STORAGE_WRITE)

    cv_file.write('stereoMapL_x',stereoMapL[0])
    cv_file.write('stereoMapL_y',stereoMapL[1])

    cv_file.write('stereoMapR_x',stereoMapR[0])
    cv_file.write('stereoMapR_y',stereoMapR[1])

    cv_file.release()

# %%
# STEREORECTIFICATION OF NOT OCLUDED INPUT VIDEO
def Rectify():
    import numpy as np
    import cv2
    import glob
    import matplotlib.pyplot as plt
    import os

    #Load images
    notoccluded_left = glob.glob('../../video/not_occluded/left/*.png')
    notoccluded_right = glob.glob('../../video/not_occluded/right/*.png')

    #assert notocluded_left, notocluded_right

    if (len(notoccluded_right)==len(notoccluded_left)):
        print("Correct database")

    num_imgs= len(notoccluded_left)

    for i in range(0,num_imgs):

        #takes each frame one by one and rectifies using right stereomap
        frame_left = cv2.imread(notoccluded_left[i])
        left_good= cv2.remap(frame_left, stereoMapL[0], stereoMapL[1], cv2.INTER_LINEAR)
        cv2.imwrite('../../Calibration/videos_rectified/not_occluded/left/leftimage'+str(i)+'.png', left_good)

        frame_right= cv2.imread(notoccluded_right[i])
        right_good= cv2.remap(frame_right, stereoMapR[0], stereoMapR[1], cv2.INTER_LINEAR)
        cv2.imwrite('../../Calibration/videos_rectified/not_occluded/right/rightimage'+str(i)+'.png', right_good)

    #Now we can use the input images from either the left or right calibrated camera

    # %%
    # STEREORECTIFICATION OF OCLUDED INPUT VIDEO

    #Load images

    occluded_left = glob.glob('../../video/occluded/left/*.png')
    occluded_right = glob.glob('../../video/occluded/right/*.png')

    if (len(occluded_right)==len(occluded_left)):
        print("Correct database")

    num_imgs= len(occluded_left)

    for i in range(0,num_imgs):

        #takes each frame one by one and rectifies using right stereomap
        frame_left = cv2.imread(occluded_left[i])
        left_good= cv2.remap(frame_left, stereoMapL[0], stereoMapL[1], cv2.INTER_LINEAR)
        cv2.imwrite('../../Calibration/videos_rectified/occluded/left/leftimage'+str(i)+'.png', left_good)

        frame_right= cv2.imread(notoccluded_right[i])
        right_good= cv2.remap(frame_right, stereoMapR[0], stereoMapR[1], cv2.INTER_LINEAR)
        cv2.imwrite('../../Calibration/videos_rectified/occluded/right/rightimage'+str(i)+'.png', right_good)


    #Now we can use the input images from either the left or right calibrated camera


