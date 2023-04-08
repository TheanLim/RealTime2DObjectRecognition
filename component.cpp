//
//  component.cpp
//  Project3
//
// This file contains main components of the project
// namely, threshold, morph, segment, obtain OOV related features,
// draw OOB and create features of different segments/components.
//  Created by Thean Cheat Lim on 2/14/23.
//

#include "component.hpp"
#include <opencv2/opencv.hpp>
#include "utils.hpp"
#include <map>
#include <string>

using namespace cv;
using namespace std;

// Given a set of points that belong to a connected component (such as a hammer),
// returns the angle of the axis of the least central moment.
// points - points of a connected component
double getAngleLeastCentralMoment(vector<Point> points){
    double cross = 0, centralY = 0, centralX=0;
    // Number of pixels in the region - RAW moment M00
    unsigned long int m00 = points.size();
    
    // Calculate Mean
    double xmean = 0;
    double ymean = 0;
    for (auto point: points){
        xmean+=point.x;
        ymean+=point.y;
    }
    xmean/=m00;
    ymean/=m00;

    // Calculate Central Moments
    for (auto point: points){
        centralY+=(point.y-ymean)*(point.y-ymean);
        centralX+=(point.x-xmean)*(point.x-xmean);
        cross+=(point.x-xmean)*(point.y-ymean);
    }
    centralY/=m00;
    centralX/=m00;
    cross/=m00;
    
    return 0.5*atan2(2*cross, centralX-centralY);
}

// Renders the specified text string in the image at a certain orientation
// img Image
// radian - Angle of orientation in radian.
// text -Text string to be drawn.
// Point - Bottom-left corner of the text string in the image.
// fontFace - Font type, see #HersheyFonts.
// fontScale - Font scale factor that is multiplied by the font-specific base size.
// color - Text color.
// thickness - Thickness of the lines used to draw a text.
// lineType - Line type. See #LineTypes
// bottomLeftOrigin - When true, the image data origin is at the bottom-left corner. Otherwise,
int putTextRotated(cv::Mat &img, double radian, const String& text, cv::Point Point, int fontFace, double fontScale, Scalar color, int thickness = 1, int lineType = LINE_8, bool bottomLeftOrigin = false){
    // Double the size because some words are OOB when place horizontal,
    // But should be inframe when rotated
    cv::Mat emptyColor = cv::Mat::zeros(img.rows, 2*img.cols, img.type());
    cv::Mat emptyWhite = cv::Mat::zeros(img.rows, 2*img.cols, img.type());
    
    cv::putText(emptyColor, text, Point, fontFace, fontScale, color, thickness, lineType, bottomLeftOrigin);
    cv::putText(emptyWhite, text, Point, fontFace, fontScale, cv::Scalar(255,255,255), thickness, lineType, bottomLeftOrigin);
    
    // Rotate text
    // Project each point onto the axis of least central moment
    // and look for smallest/largest x and y values
    double cosRadian = cos(radian);
    double sinRadian = sin(radian);
    for(int i=0; i<emptyWhite.rows; i++){
        cv::Vec3b *wptr = emptyWhite.ptr<cv::Vec3b>(i);
        cv::Vec3b *cptr = emptyColor.ptr<cv::Vec3b>(i);
        for(int j=0; j<emptyWhite.cols; j++){
            // Skip non words area
            if (wptr[j][0]!=0){
                int x_rotated = (j-Point.x)*cosRadian - (i-Point.y)* sinRadian+Point.x;
                int y_rotated = (j-Point.x)*sinRadian + (i-Point.y)* cosRadian+Point.y;
                if (0<=x_rotated && x_rotated<img.cols && 0<=y_rotated && y_rotated<img.rows){
                    for (int c = 0; c<3; c++){
                        img.at<cv::Vec3b>(y_rotated, x_rotated)[c] =
                            img.at<cv::Vec3b>(y_rotated, x_rotated)[c] - wptr[j][c] + cptr[j][c];
                    }
                }
            }
        }
    }
    return 0;
}

// Construct Oriented Bounding Box given a vector of points and orientation of interest.
// points - Points of a region/component which should be used to construct OBB.
// radian - Angle of orientation in radian
// OBBpoints - Points for Oriented Bounding Box including the four courners, center of box and those for drawing
//      the line of the axis of least central moment
// stats - Statistics related to the bounding box such as height-to-width ratio
int getOOB(vector<Point> points, double radian, vector<Point> &OBBpoints, vector<float>&stats){
    // Rotate clockwise at angle 'radian'
    // Initialize the minimum and maximum x and y values
    float min_x = FLT_MAX;
    float max_x = FLT_MIN;
    float min_y = FLT_MAX;
    float max_y = FLT_MIN;
    
    // Calculate Mean
    double xmean = 0;
    double ymean = 0;
    for (auto point: points){
        xmean+=point.x;
        ymean+=point.y;
    }
    xmean/=points.size();
    ymean/=points.size();
    
    // Project each point onto the axis of least central moment
    // and look for smallest/largest x and y values
    double cosRadian = cos(radian);
    double sinRadian = sin(radian);
    for (auto point : points) {
        // Project the point onto the line of the given angle
        /*https:danceswithcode.net/engineeringnotes/rotations_in_2d/rotations_in_2d.html*/
        // x1 = (x0 – xc)cos(θ) – (y0 – yc)sin(θ) + xc
        // y1 = (x0 – xc)sin(θ) + (y0 – yc)cos(θ) + yc
        float x_on_line = (point.x-xmean)*cosRadian - (point.y-ymean)* sinRadian+xmean;
        float y_on_line = (point.x-xmean)*sinRadian + (point.y-ymean)* cosRadian+ymean;
        
        // Update the min and max values for both axis
        min_x = std::min(min_x, x_on_line);
        max_x = std::max(max_x, x_on_line);
        min_y = std::min(min_y, y_on_line);
        max_y = std::max(max_y, y_on_line);
    }
    
    int width = max_x-min_x;
    int height =max_y-min_y;
    OBBpoints.push_back(cv::Point(min_x, min_y));
    OBBpoints.push_back(cv::Point(min_x, max_y));
    OBBpoints.push_back(cv::Point(max_x, max_y));
    OBBpoints.push_back(cv::Point(max_x, min_y));
    // Axis of rotation and its perpendicular
    OBBpoints.push_back(cv::Point(max_x, ymean));
    OBBpoints.push_back(cv::Point(xmean, ymean));
    OBBpoints.push_back(cv::Point(xmean, min_y));

    // Convert coordinates back to the orginal coordinates for plotting
    double cosRadianNeg = cos(-radian);
    double sinRadianNeg = sin(-radian);
    for (int i = 0; i< OBBpoints.size(); i++){
        int x = (OBBpoints[i].x-xmean)*cosRadianNeg - (OBBpoints[i].y-ymean)* sinRadianNeg+xmean;
        int y = (OBBpoints[i].x-xmean)*sinRadianNeg + (OBBpoints[i].y-ymean)* cosRadianNeg+ymean;
        OBBpoints[i].x = x;
        OBBpoints[i].y = y;
    }
    
    stats.push_back(width);
    stats.push_back(height);
    stats.push_back(height*1.0f/width);
    stats.push_back(points.size()/(width*height*1.0f));
    return 0;
}

// Threshold the input/source image and make it a binary image
// Use K-Means clustering to determine the threshold value
// between foreground and background image
// src - Source image
// dst - Destination/output image
// threshold - threshold value
int threshold(cv::Mat &src, cv::Mat&dst){
    // Gaussian Blur images
    cv::Mat blurred;
    blur5x5(src, blurred);
    
    // Making high saturation value pixels darker
    cv::Mat darken;
    darkerHighSaturation(blurred, darken, 255/2, 2.0);
    
    // Convert to grayscale for Kmeans clustering
    cv::Mat gray;
    cv::cvtColor(darken, gray, COLOR_BGR2GRAY);
    cv::Mat data = gray.reshape(1, gray.rows * gray.cols);
    cv::Mat dataKnn;
    data.convertTo(dataKnn, CV_32F);

    // K-means parameters
    int k = 2, attempts = 1, max_iter = 50;
    double eps = 1.0;
    cv::Mat labels, centroids;
    cv::TermCriteria criteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, max_iter, eps);
    cv::kmeans(dataKnn, k, labels, criteria, attempts, cv::KMEANS_PP_CENTERS, centroids);

    // Calculate the average of the two centroids
    cv::Scalar c1 = mean(centroids.row(0));
    cv::Scalar c2 = mean(centroids.row(1));
    cv::Scalar avg_centroid = (c1 + c2) / 2.0;
    int threshold = avg_centroid[0];
    
    cv::cvtColor(gray, dst, COLOR_GRAY2BGR);
    // Set to 255 if above threshold, 0 if below threshold
    for(int i=0; i<dst.rows; i++){
        cv::Vec3b *dptr = dst.ptr<cv::Vec3b>(i);
        for(int j=0; j<dst.cols; j++){
            if (dptr[j][0] < threshold){
                dptr[j][0] = 255;
                dptr[j][1] = 255;
                dptr[j][2] = 255;
            } else {
                dptr[j][0] = 0;
                dptr[j][1] = 0;
                dptr[j][2] = 0;
            }
        }
    }
    return 0;
}

// Apply morphological filters onto the input/source image
// src - Source image
// dst - Destination/output image
int morphFilter(cv::Mat &src, cv::Mat&dst){
    /*https:docs.opencv.org/3.4/db/df6/tutorial_erosion_dilatation.html*/
    cv::Mat fourNeigh = cv::getStructuringElement(cv::MORPH_CROSS, cv::Size(3, 3));
    cv::Mat eightNeigh = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
    
    // Erosion than Dilation -- remove noise
    cv::morphologyEx(src, dst, cv::MORPH_OPEN, fourNeigh, cv::Point(-1,-1), 2);
    // Dilation followed by Erosion -- closing small holes
    cv::morphologyEx(dst, dst, cv::MORPH_CLOSE, fourNeigh, cv::Point(-1,-1), 2);

    return 0;
}

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
int segment(cv::Mat &src, cv::Mat&dst, bool color, int topNSegment, std::vector<int> &largestAreaLabels, map<int, std::vector<cv::Point>> &component_points, cv::Mat &labels, cv::Mat &stats, cv::Mat &centroids){
    cv::Mat grayImg;
    cv::cvtColor(src, grayImg, cv::COLOR_BGR2GRAY);
    
    int n_components = cv::connectedComponentsWithStats(grayImg, labels, stats, centroids);
    
    // Collect and sort component areas descendingly
    std::vector<std::pair<int, int>> areas;
    for (int i = 0; i < n_components; i++)
    {
        // at least area of 100
        if (stats.at<int>(i, cv::CC_STAT_AREA) > 100){
            areas.push_back(std::pair(stats.at<int>(i, cv::CC_STAT_AREA), i));
        }
    }
    std::sort(areas.rbegin(), areas.rend());
    std::vector<cv::Vec3b> colorLabels;
    for (int i = 0; i < n_components; i++)
    {
        largestAreaLabels.push_back(areas[i].second);
        // Set the same color for each component
        colorLabels.push_back(cv::Vec3b((i+1)*100 % 256, (i+1)*200 % 256, (i+1)*300 % 256));
    }
    
    // Color topNSegment connected components in the image
    if (color) src.copyTo(dst);
    for (int i = 0; i < src.rows; i++) {
        for (int j = 0; j < src.cols; j++) {
            int label = labels.at<int>(i, j);
            /*https:stackoverflow.com/questions/24139428/check-if-element-is-in-the-list-contains*/
            bool found = (std::find(largestAreaLabels.begin(), largestAreaLabels.begin()+topNSegment+1, label) != largestAreaLabels.begin()+topNSegment+1);
            if (found and label >0) {  //label 0 is the biggest background - skip it
                if (color) dst.at<cv::Vec3b>(i, j) = colorLabels[label];
                component_points[label].push_back(cv::Point(j, i));
            }
        }
    }
    return 0;
}

// Compute Oriented Bounding Box related features
// component_points - a map of label to Point(x,y) of the source image from the topN regions
// OOB_points -  Points for Oriented Bounding Box including the four courners, center of box and those for drawing
//      the line of the axis of least central moment
// OOB_stats - Statistics related to the bounding box such as height-to-width ratio
// OOB_angles - Angle of orientation of the bounding boxes, in radian
int OOBFeatures(map<int, std::vector<cv::Point>> &component_points, std::vector<std::vector<cv::Point>> &OOB_points, std::vector<std::vector<float>>&OOB_stats, std::vector<double>&OOB_angles){
    // Draw a oriented bounding box around topNSegment connected component
    for (auto const& [key, val] : component_points)
    {
        double angle = getAngleLeastCentralMoment(val);
        OOB_angles.push_back(angle);
        std::vector<cv::Point> temp_OOB_points;
        std::vector<float> temp_OOBstats;
        // -angle because we want to counter the angle
        getOOB(val, -angle,temp_OOB_points, temp_OOBstats);
        OOB_points.push_back(temp_OOB_points);
        OOB_stats.push_back(temp_OOBstats);
    }
    return 0;
}

// Draw bounding boxes onto an input image
// img - Image to draw bounding boxes on
// OOB_points - Points for Oriented Bounding Box including the four courners, center of box and those for drawing
//      the line of the axis of least central moment
// OOB_stats - Statistics related to the bounding box such as height-to-width ratio
// OOB_angles - Angle of orientation of the bounding boxes, in radian
// OOB_labels - labels/class texts to attach to each bounding box
int drawOOB(cv::Mat &img, std::vector<std::vector<cv::Point>> &OOB_points, std::vector<std::vector<float>>&OOB_stats, std::vector<double>&OOB_angles, std::vector<string> &OOB_labels){
    // Drawing the oriented Bounding Box
    for (int counter = 0; counter<OOB_points.size(); counter ++){
        // The first four are the box
        for (int i = 0; i < 4; i++) {
            cv::line(img, OOB_points[counter][i], OOB_points[counter][(i + 1) % 4], cv::Scalar(255, 0, 0), 2);
        }
        // The next three are for the central axis
        for (int i = 4; i < 7; i++) {
            cv::line(img, OOB_points[counter][i], OOB_points[counter][(i + 1) % 7], cv::Scalar(255, 0, 0), 2);
        }

        putTextRotated(img,
                    OOB_angles[counter],
                    OOB_labels[counter],
                    OOB_points[counter][0],
                    cv::FONT_HERSHEY_DUPLEX,
                    std::max(OOB_stats[counter][0], OOB_stats[counter][1])*1.0/std::max(img.rows, img.cols)*5,
                    CV_RGB(118, 185, 110), //font color
                    2.5,
                    LINE_4);
    }
    return 0;
}

// Create features
// OOB_stats - Statistics related to the bounding box such as height-to-width ratio
// labels - labels of segmented regions
// largestAreaLabels - labels of the top N largest regions
// featureVec - feature Vector. This is where computed features are populated
int featurize(std::vector<std::vector<float>>&OOB_stats, cv::Mat &labels, std::vector<int> &largestAreaLabels, std::vector<std::vector<float>> &featureVec){
    
    /*https:stackoverflow.com/questions/47537049/what-is-the-best-way-to-sort-a-vector-leaving-the-original-one-unaltered*/
    vector<int> largestAreaLabels_sorted(largestAreaLabels.size());
    partial_sort_copy(begin(largestAreaLabels), end(largestAreaLabels), begin(largestAreaLabels_sorted), end(largestAreaLabels_sorted));
    
    for (int i = 0; i<OOB_stats.size(); i++){
        std::vector<float> temp;
        // skip the first two stats - width and height
        for (int j = 2; j<OOB_stats[i].size(); j++){
            temp.push_back(OOB_stats[i][j]);
        }
        
        cv::Moments moments = cv::moments(labels == largestAreaLabels_sorted[i],true);
        double hu[7];
        cv::HuMoments(moments, hu);
        for (int i = 0; i<1; i++){
            double logHu = -1 * copysign(1.0, hu[i]) * log10(abs(hu[i]));
            temp.push_back(logHu);
        }
        featureVec.push_back(temp);
    }
    return 0;
}
