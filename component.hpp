//
//  component.hpp
//  Project3
//
// This file contains main components of the project
// namely, threshold, morph, segment, obtain OOV related features,
// draw OOB and create features of different segments/components.
//  Created by Thean Cheat Lim on 2/14/23.
//

#ifndef component_hpp
#define component_hpp

#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <map>
#include <string>

// Threshold the input/source image and make it a binary image
// Use K-Means clustering to determine the threshold value
// between foreground and background image
// src - Source image
// dst - Destination/output image
// threshold - threshold value
int threshold(cv::Mat &src, cv::Mat&dst);

// Apply morphological filters onto the input/source image
// src - Source image
// dst - Destination/output image
int morphFilter(cv::Mat &src, cv::Mat&dst);

// Segment the source image and color the top N largest regions
// src - Source image
// dst  - Destination/output image
// color - Boolean. To color the top N segments/regions/components or not
// topNSegment - Number of N largest regions to keep
// largestAreaLabels - labels of the top N largest regions
// component_points - a map of label to Point(x,y) of the source image from the topN regions
// labels - destination labeled image
// stats - statistics output for each label, including the background label.
// centroids - centroid output for each label, including the background label.
int segment(cv::Mat &src, cv::Mat&dst, bool color, int topNSegment, std::vector<int> &largestAreaLabels, std::map<int, std::vector<cv::Point>> &component_points, cv::Mat &labels, cv::Mat &stats, cv::Mat &centroids);

// Compute Oriented Bounding Box related features
// component_points - a map of label to Point(x,y) of the source image from the topN regions
// OOB_points -  Points for Oriented Bounding Box including the four courners, center of box and those for drawing
//      the line of the axis of least central moment
// OOB_stats - Statistics related to the bounding box such as height-to-width ratio
// OOB_angles - Angle of orientation of the bounding boxes, in radian
int OOBFeatures(std::map<int, std::vector<cv::Point>> &component_points, std::vector<std::vector<cv::Point>> &OOB_points, std::vector<std::vector<float>>&OOB_stats, std::vector<double>&OOB_angles);

// Draw bounding boxes onto an input image
// img - Image to draw bounding boxes on
// OOB_points - Points for Oriented Bounding Box including the four courners, center of box and those for drawing
//      the line of the axis of least central moment
// OOB_stats - Statistics related to the bounding box such as height-to-width ratio
// OOB_angles - Angle of orientation of the bounding boxes, in radian
// OOB_labels - labels/class texts to attach to each bounding box
int drawOOB(cv::Mat &img, std::vector<std::vector<cv::Point>> &OOB_points, std::vector<std::vector<float>>&OOB_stats, std::vector<double>&OOB_angles, std::vector<std::string> &OOB_labels);

// Create features
// OOB_stats - Statistics related to the bounding box such as height-to-width ratio
// labels - labels of segmented regions
// largestAreaLabels - labels of the top N largest regions
// featureVec - feature Vector. This is where computed features are populated
int featurize(std::vector<std::vector<float>>&OOB_stats, cv::Mat &labels,  std::vector<int> &largestAreaLabels, std::vector<std::vector<float>> &featureVec);
#endif /* component_hpp */
