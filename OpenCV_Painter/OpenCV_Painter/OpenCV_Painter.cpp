#include "pch.h"
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <math.h>


using namespace std;
using namespace cv;


//int main(int)
//{
//	VideoCapture cap(0); // open the default camera
//	if (!cap.isOpened())  // check if we succeeded
//		return -1;
//
//	Mat edges;
//	namedWindow("edges", 1);
//	for (;;)
//	{
//		Mat frame;
//		cap >> frame; // get a new frame from camera
//		cvtColor(frame, edges, COLOR_BGR2GRAY);
//		GaussianBlur(edges, edges, Size(7, 7), 1.5, 1.5);
//		Canny(edges, edges, 0, 30, 3);
//		imshow("edges", edges);
//		if (waitKey(30) >= 0) break;
//	}
//	// the camera will be deinitialized automatically in VideoCapture destructor
//	return 0;
//}



int main() {
	VideoCapture cap(0);		//Open Camera
	

	namedWindow("Frame", 1);
	namedWindow("Processed Frame", 1);
		for (;;)
		{
			Mat frame;
			Mat processedFrame;
				
			cap >> frame;													// Get a new frame from camera
			flip(frame, frame, 1);											// Flip frame left-right
			processedFrame = frame.clone();									// Dissasociativley copy frame data to proccessing container
			GaussianBlur(processedFrame, processedFrame, Size(11, 11), 0);	// Apply a pretty strong blur
			cvtColor(processedFrame, processedFrame, COLOR_BGR2HSV);		// Move to HSV colorspace
			
			inRange(processedFrame, Scalar(80, 100, 30), Scalar(127, 255, 255), processedFrame);
			erode(processedFrame, processedFrame, 2);
			dilate(processedFrame, processedFrame, 2);

			vector<vector<Point> > contours;
			vector<Vec4i> hierarchy;
			
			findContours(processedFrame, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

			int largestContourIdx = 0;
			double largestContourArea = 0.0;
			double a = 0.0;

			for (int i = 0; i < contours.size(); i++) {
				a = contourArea(contours[i], false);
				if (a > largestContourArea) {
					largestContourArea = a;
					largestContourIdx = i;
				}
			}

			drawContours(frame, contours, largestContourIdx, Scalar(255, 0, 0), 3);

			//vector<vector<Point> > contours_poly(contours.size());
			vector<Point> contours_poly; 
			Point2f centers;
			float radius;
			
			approxPolyDP(contours[largestContourIdx], contours_poly, 3, true);
			minEnclosingCircle(contours_poly, centers, radius);
			circle(frame, centers, (int)radius, Scalar(255, 0, 0), 2);
			circle(frame, centers, 2, Scalar(255, 0, 0), 4);



			imshow("Frame", frame);
			imshow("Processed Frame", processedFrame);
			if (waitKey(30) >= 0) break;
		}




	return 0;
}
