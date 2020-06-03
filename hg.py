#Author: Sardor Allaberganov
#ID: U1610202
#Group CSE 16-2

#------------------------------------------------
# Project Name: HAND AND FINGER DETECTION CONTROL
#------------------------------------------------

#importing opencv library for image processing
import cv2 as cv
#importing numpy package for scientific computing with Python 
import  numpy as np
# sklearn.metrics module includes score functions, performance metrics and pairwise metrics and distance computations
from sklearn.metrics import pairwise

#global variables
background = None

#---------------------------------------------------------
# Funtion to find the running average over the background
#---------------------------------------------------------
def average_bg(image, aWeighted):
    global background
    #initialize background
    if background is None:
        background = image.copy().astype("float")
        return
    
    #opencv function to compute weighted average image, accumulate it and update the background
    cv.accumulateWeighted(image, background, aWeighted)

#----------------------------------------------------
# Funtion to segment the region of hand in the image
#----------------------------------------------------
def hand_segment(image):
    global background
    
    #find the absolute difference between background and current frame
    difference = cv.absdiff(background.astype("uint8"), image)

    #threshold the difference image so that we get the foreground
    thresholded = cv.threshold(difference, 12, 255, cv.THRESH_BINARY)[1]

    #get the contours in the thresholded image
    contours, hierarchy = cv.findContours(thresholded.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    #check whether contours is detected or not. return None if no contours
    if len(contours) ==0:
    	return
    else: 
    	# based on contour area, get the maximum contour which is the hand
    	area = cv.contourArea
    	detected_segment = max(contours, key = area)

    	return (thresholded, detected_segment)


#--------------------------------------------------------------
# To count the number of fingers in the segmented hand region
#--------------------------------------------------------------
def count_finger(thresholded, detected_segment):
	#find to convex hull of the detected segment of hand region
	convex = cv.convexHull(detected_segment)

	#find the most extreme points in the convex hull
	e_top = tuple(convex[convex[:, :, 1].argmin()][0]) 
	e_bottom = tuple(convex[convex[:, :, 1].argmax()][0]) 
	e_left = tuple(convex[convex[:, :, 0].argmin()][0]) 
	e_right = tuple(convex[convex[:, :, 0].argmax()][0]) 

	#find the center of the palm
	cX = int((e_left[0] + e_right[0]) / 2)
	cY = int((e_top[1] + e_bottom[1]) / 2)

    # find the maximum euclidean distance between the center of the palm
    # and the most extreme points of the convex hull
	distance = pairwise.euclidean_distances([(cX, cY)], Y=[e_left, e_right, e_top, e_bottom])[0]
	maximum_distance = distance[distance.argmax()]

     # calculate the radius of the circle with 80% of the max euclidean distance obtained
	radius = int(0.7 * maximum_distance)

    # find the circumference of the circle
	circumference = (2 * np.pi * radius)

    # take out the circular region of interest which has 
    # the palm and the fingers
	circular_roi = np.zeros(thresholded.shape[:2], dtype="uint8")
	
    # draw the circular ROI
	cv.circle(circular_roi, (cX, cY), radius, 255, 1)

    # take bit-wise AND between thresholded hand using the circular ROI as the mask
    # which gives the cuts obtained using mask on the thresholded hand image
	circular_roi = cv.bitwise_and(thresholded, thresholded, mask=circular_roi)

    # compute the contours in the circular ROI
	(contour, hierarchy) = cv.findContours(circular_roi.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

    # initalize the finger count
	count = 0

    # loop through the contours found
	for c in contour:
        # compute the bounding box of the contour
		(x, y, w, h) = cv.boundingRect(c)

        # increment the count of fingers only if -
        # 1. The contour region is not the wrist (bottom area)
        # 2. The number of points along the contour does not exceed
        #     25% of the circumference of the circular ROI
		if ((cY + (cY * 0.25)) > (y + h)) and ((circumference * 0.25) > c.shape[0]):
			count += 1

	return count

#-----------------
# main function
#-----------------
if __name__ == "__main__":
    # initialize accumulated weight
    accumWeight = 0.5

    # get the reference to the webcam
    video = cv.VideoCapture(0)

    # region of interest (ROI) coordinates
    top, right, bottom, left = 0, 400, 225, 630

    # initialize num of frames
    num_frames = 0

    # calibration indicator
    calibrated = False

    # keep looping, until interrupted
    while(True):
        # get the current frame
        (grabbed, frame) = video.read()

        # flip the frame so that it is not the mirror view
        frame = cv.flip(frame, 1)

        # clone the frame
        clone = frame.copy()

        # get the height and width of the frame
        (height, width) = frame.shape[:2]

        # get the ROI
        roi = frame[top:bottom, right:left]
        cv.imshow("ROI", roi)

        # convert the roi to grayscale and blur it
        gray = cv.cvtColor(roi, cv.COLOR_BGR2GRAY)
        cv.imshow("Gray image", gray)
        gray = cv.GaussianBlur(gray, (7, 7), 0)
        cv.imshow("Blurred image", gray)

        # to get the background, keep looking till a threshold is reached
        # so that our weighted average model gets calibrated
        if num_frames < 30:
            average_bg(gray, accumWeight)
        else:
            # segment the hand region
            hand = hand_segment(gray)

            # check whether hand region is segmented
            if hand is not None:
                # if yes, unpack the thresholded image and
                # segmented region
                (thresholded, segmented) = hand

                # draw the segmented region and display the frame
                cv.drawContours(clone, [segmented + (right, top)], -1, (0, 0, 255))

                #count the number of fingers
                fingers = count_finger(thresholded, segmented)

                if (fingers == 0):
                	cv.putText(clone, str("Stop song"), (70, 75), cv.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)
                elif(fingers == 1):
                	cv.putText(clone, str("Play song"), (70, 75), cv.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)
                elif(fingers == 2):
                	cv.putText(clone, str("Prev song"), (70, 75), cv.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)
                elif(fingers == 3):
                	cv.putText(clone, str("Next song"), (70, 75), cv.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)
                elif(fingers == 4):
                	cv.putText(clone, str("Sound up"), (70, 75), cv.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)
                elif(fingers == 5):
                	cv.putText(clone, str("Sound down"), (70, 75), cv.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)
                else:
                	print("Unknown command")
                #put text to the showing window
                cv.putText(clone, str(fingers), (70, 45), cv.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)
                
                # show the thresholded image
                cv.imshow("Thesholded", thresholded)

        # draw the segmented hand
        cv.rectangle(clone, (left, top), (right, bottom), (0,255,0), 2)

        # increment the number of frames
        num_frames += 1

        # resize the frame
        clone = cv.resize(clone, (1024,768))

		# display the frame with segmented hand
        cv.imshow("Video Feed", clone)

        # observe the keypress by the user
        keypress = cv.waitKey(1) & 0xFF

        # if the user pressed "q", then stop looping
        if keypress == ord("q"):
            break


    #code to test with image 
    
	# import numpy as np
	# import cv2 as cv

	# src = cv.imread('hand.jpeg')

	# im = cv.resize(src, (500,600))

	# imgray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
	# thresh = cv.threshold(imgray, 245, 255, cv.THRESH_BINARY)[1]
	# contours, hierarchy = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

	# image = cv.drawContours(thresh, contours, -1, (0,255,0), 3)
	# cv.imshow("result", image)

	# cv.waitKey(0)
