//
//  utils.hpp
//  Project3
//  This file contains utility functions.
//
//  Created by Thean Cheat Lim on 2/14/23.
//

#ifndef utils_hpp
#define utils_hpp

#include <stdio.h>
#include <opencv2/opencv.hpp>

// Blurs the input image using a 5x5 Gaussian filter
// src - source (input) image
// dst - destination (output) image
int blur5x5(cv::Mat &src, cv::Mat &dst);

// Darken pixels with high saturation values.
// Convert the input image into HSV space,
// and for each pixel that has its saturation value
// larger than the `saturationThreshold`, multiply
// the value (of HSV) by the `factor`
// src - source (input) image
// dst - destination (output) image
// saturationThreshold - saturation threshold
// factor - factor of multiplying the value
int darkerHighSaturation(cv::Mat &src, cv::Mat &dst, int saturationThreshold, double factor);

// Save the given frames
// frame - a video frame/image
// effectFrame - a video frame/image with effects applied
int saveFrames(cv::Mat &frame, cv::Mat &effectFrame);

/*https:www.johndcook.com/blog/standard_deviation/*/
// A Class to calculate running statistics efficiently
class RunningStat
    {
    public:
        RunningStat();
        // Clear the running statistics
        void Clear();
        // Push a new number and calculate the new running statistics
        void Push(double x);
        int NumDataValues() const;
        // Retrieve the mean
        double Mean() const;
        // Retrieve the Variance
        double Variance() const;
        // Retrieve the StandardDeviation
        double StandardDeviation() const;
    private:
        int m_n;
        double m_oldM, m_newM, m_oldS, m_newS;
};


#endif /* utils_hpp */
