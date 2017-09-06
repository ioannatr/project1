from collections import deque
import numpy as np
import argparse
import imutils
import cv2, matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


class Vision(object):
	#initialising the boundaries of the colour we want to detect and the frame
	def __init__(self, upper_bound, lower_bound):
		self.upper_bound = upper_bound
		self.lower_bound = lower_bound
		
	def threshold(self, hsv):
		# construct a mask for the color "green", then perform
		# a series of dilations and erosions to remove any small
		# blobs left in the mask
		mask = cv2.inRange(hsv, self.lower_bound, self.upper_bound)
		mask = cv2.erode(mask, None, iterations=2)
		mask = cv2.dilate(mask, None, iterations=2)
		
		return mask
	
	def detect(self,mask, frame, pts):
		# perform compute the contour (i.e. outline) of the green ball and draw it on our frame
		# find contours in the mask and initialize the current
		# (x, y) center of the ball
		cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
			cv2.CHAIN_APPROX_SIMPLE)[-2]
		
		center = None
	
		# only proceed if at least one contour was found
		if len(cnts) > 0:
			# find the largest contour in the mask, then use
			# it to compute the minimum enclosing circle and
			# centroid
			c = max(cnts, key=cv2.contourArea)
			((x, y), radius) = cv2.minEnclosingCircle(c)
			print((x, y), radius)
			M = cv2.moments(c)
			center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
			# only proceed if the radius meets a minimum size
			if radius > 10:
				# draw the circle and centroid on the frame,
				# then update the list of tracked points
				cv2.circle(frame, (int(x), int(y)), int(radius),
					(0, 255, 255), 2)
				cv2.circle(frame, center, 5, (0, 0, 255), -1)
			
			#############################################################
			###pws na to kanw n epistrepsei tipota na den vriskei kati###
			#############################################################
			
			
			return center
		
	def draw(self, frame , pts, args):
	
		#last step.Draw the contrail of the ball
		# loop over the set of tracked points
		for i in range(1, len(pts)):
			# if either of the tracked points are None, ignore
			# them
			if pts[i - 1] is None or pts[i] is None:
				continue
	
			# otherwise, compute the thickness of the line and
			# draw the connecting lines
			thickness = int(np.sqrt(args["buffer"] / float(i + 1)) * 2.5)
			cv2.line(frame, pts[i - 1], pts[i], (0, 0, 255), thickness)
			# show the frame to our screen
			
		#cv2.imshow("Frame", frame)
		#key = cv2.waitKey(1) & 0xFF
		plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
		plt.show()
		#if the 'q' key is pressed, stop the loop
		#if key == ord("q"):
		#	break