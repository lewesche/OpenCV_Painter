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
			//
			//Point2f center;
			//float radius;
			//minEnclosingCircle(contours_poly[largestContourIdx], center, radius);


			vector<vector<Point> > contours_poly(contours.size());
			vector<Rect> boundRect(contours.size());
			vector<Point2f>centers(contours.size());
			vector<float>radius(contours.size());
			for (size_t i = 0; i < contours.size(); i++)
			{
				approxPolyDP(contours[i], contours_poly[i], 3, true);
				boundRect[i] = boundingRect(contours_poly[i]);
				minEnclosingCircle(contours_poly[i], centers[i], radius[i]);
				circle(frame, centers[i], (int)radius[i], Scalar(255, 0, 0), 2);
			}



			imshow("Frame", frame);
			imshow("Processed Frame", processedFrame);
			if (waitKey(30) >= 0) break;
		}




	return 0;
}
