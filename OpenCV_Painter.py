import numpy as np
import cv2
import time
import argparse
import imutils
from imutils.video import VideoStream
from collections import deque

########################################################
#Enter colors, last colors show up on top
#Choose from green, blue, red, yellow, purple        
colors = ["red", "blue", "green"]
trail_length = 30 # Determines the length of the path
########################################################

ind = len(colors)
boundLower = [] #HSV colors
boundUpper = [] #HSV colors
markerColor = [] #BGR colors
pts = [] #Container for trail points
colorLabel = [];

#Save color constants
if "green" in colors:
	colorLabel.append("green")
	greenLower = (40, 110, 80)
	greenUpper = (70, 255, 225)
	greenMarker = (0, 255, 0)
	greenPts = deque(maxlen=trail_length)
	boundLower.append(greenLower)
	boundUpper.append(greenUpper)
	markerColor.append(greenMarker)
	pts.append(greenPts)

if "blue" in colors:
	colorLabel.append("blue")
	blueLower = (80, 100, 30)
	blueUpper = (127, 255, 255)
	blueMarker = (255, 0, 0)
	bluePts = deque(maxlen=trail_length)
	boundLower.append(blueLower)
	boundUpper.append(blueUpper)
	markerColor.append(blueMarker)
	pts.append(bluePts)

if "red" in colors:
	colorLabel.append("red")
	redLower = (80, 90, 110)  #Cyan upper and lower bounds, since image is inverted for red
	redUpper = (100, 255, 255) 
	redMarker = (0, 0, 255)
	redPts = deque(maxlen=trail_length)
	boundLower.append(redLower)
	boundUpper.append(redUpper)
	markerColor.append(redMarker)
	pts.append(redPts)

if "purple" in colors:
	colorLabel.append("purple")
	purpleLower = (140, 80, 80)
	purpleUpper = (165, 255, 225)
	purpleMarker = (255, 0, 200)
	purplePts = deque(maxlen=trail_length)
	boundLower.append(purpleLower)
	boundUpper.append(purpleUpper)
	markerColor.append(purpleMarker)
	pts.append(purplePts)

if "yellow" in colors:
	colorLabel.append("yellow")
	yellowLower = (20, 110, 100)
	yellowUpper = (30, 255, 255)
	yellowMarker = (0, 255, 255)
	yellowPts = deque(maxlen=trail_length)
	boundLower.append(yellowLower)
	boundUpper.append(yellowUpper)
	markerColor.append(yellowMarker)
	pts.append(yellowPts)

#Access on webcam
vs = VideoStream(src=0).start()
time.sleep(1.0)

#Main loop
while True:
	#Read webcam frame, flip it
	frame = np.fliplr(vs.read())

	contourView = np.fliplr(vs.read())

	if frame is None:
		print("Could not grab frame")
		break
	# resize, blur, convert to HSV
	frame = imutils.resize(frame, width=800)
	contourView = imutils.resize(contourView, width=800)
	blurred = cv2.GaussianBlur(frame, (11, 11), 0)
	hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
	
	for i in range(0, ind):
		# construct a mask for each color 
		# dilate and erode to remove imperfections

		# If the color is red, invert the convert the BGR image and look for cyan
		if colorLabel[i] == "red":
			blurredInv = ~blurred
			hsvInv = cv2.cvtColor(blurredInv, cv2.COLOR_BGR2HSV)
			mask = cv2.inRange(hsvInv, boundLower[i], boundUpper[i]) 
		else:
			mask = cv2.inRange(hsv, boundLower[i], boundUpper[i])
			
		mask = cv2.erode(mask, None, iterations=2)
		mask = cv2.dilate(mask, None, iterations=2)

		
		# find contours in the mask and initialize the current (x, y) center of each contour
		conts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
		conts = conts[0] if imutils.is_cv2() else conts[1]
		center = None

		# proceed if at least one contour was found
		if len(conts) > 0:
			# Find the largest contour
			# Enscribe a circle
			# Find the centroid
			c = max(conts, key=cv2.contourArea)

			cv2.drawContours(contourView, c, -1, markerColor[i], 3)

			((x, y), radius) = cv2.minEnclosingCircle(c)
			M = cv2.moments(c)
			center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

			# only proceed if the radius is at least 10 pixels
			if radius > 35:
				# draw bounding circle and centroid on the frame,
				cv2.circle(frame, (int(x), int(y)), int(radius), markerColor[i], 2)
				cv2.circle(frame, center, 5, markerColor[i], -1)
		
				# update the list of tracked points
				pts[i].appendleft(center)

		# loop over the set of tracked points
		for j in range(1, len(pts[i])):
			# if either of the tracked points are None, ignore them
			if pts[i][j - 1] is None or pts[i][j] is None:
				continue
			# otherwise, compute the thickness of the line and
			# draw the connecting lines
			thickness = int(np.sqrt(trail_length / float(j + 1)) * 2.5)
			cv2.line(frame, pts[i][j - 1], pts[i][j], markerColor[i], thickness)

	# show the frame to our screen
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	cv2.imshow("Contour View", contourView)
	key = cv2.waitKey(1) & 0xFF

	# if the 'q' key is pressed, stop the loop
	if key == ord("q"):
		break

# Stop the webcam, close window
vs.stop()
cv2.destroyAllWindows()
