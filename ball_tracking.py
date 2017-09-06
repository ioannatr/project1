# import the necessary packages
#pigi: http://www.pyimagesearch.com/2015/09/14/ball-tracking-with-opencv/
from vision2 import Vision
from collections import deque
import numpy as np
import argparse
import imutils
import cv2, matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", #anoigoume tin kamera
	help="path to the (optional) video file")
ap.add_argument("-b", "--buffer", type=int, default=64,
	help="max buffer size")
args = vars(ap.parse_args())

# define the lower and upper boundaries of the "green"
# ball in the HSV color space, then initialize the
# list of tracked points
greenLower = (29, 86, 6)
greenUpper = (64, 255, 255)
pts = deque(maxlen= args["buffer"])


 
# if a video path was not supplied, grab the reference
# to the webcam
if not args.get("video", False):
	camera = cv2.VideoCapture(0)
# otherwise, grab a reference to the video file
else:
	camera = cv2.VideoCapture(args["video"])

vision = Vision(greenUpper, greenLower)
# keep looping
while True:
	# grab the current frame
	(grabbed, frame) = camera.read()
 
	# if we are viewing a video and we did not grab a frame,
	# then we have reached the end of the video
	if args.get("video") and not grabbed:
		break
 
	# resize the frame, blur it, and convert it to the HSV
	# color space
	frame = imutils.resize(frame, width=600)
	# blurred = cv2.GaussianBlur(frame, (11, 11), 0)
	hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
	
	mask = vision.threshold(hsv)
	
	center =vision.detect(mask, frame, pts)
	# update the points queue
	pts.appendleft(center)
	
	vision.draw(frame, pts, args)
# cleanup the camera and close any open windows
camera.release()
cv2.destroyAllWindows()