//
//  utils.cpp
//  Project3
//  This file contains utility functions.
//
//  Created by Thean Cheat Lim on 2/14/23.
//

#include "utils.hpp"
#include <opencv2/opencv.hpp>

// Blurs the input image using a 5x5 Gaussian filter
// src - source (input) image
// dst - destination (output) image
int blur5x5(cv::Mat &src, cv::Mat &dst){
    // Filter [1 2 4 2 1]
    src.copyTo(dst);  // Keep the orig values as if if not modified
    
    // Row 1D
    for(int i=0; i<src.rows; i++){
        cv::Vec3b *sptr = src.ptr<cv::Vec3b>(i);
        // Destination pointer
        cv::Vec3b *dptr = dst.ptr<cv::Vec3b>(i);
        for(int j=2; j<src.cols-2; j++){
            for(int c=0;c<3;c++){
                dptr[j][c] =
                (
                 1*sptr[j-2][c]
                 +2*sptr[j-1][c]
                 +4*sptr[j][c]
                 +2*sptr[j+1][c]
                 +1*sptr[j+2][c]
                )/10;
            }
        }
    }
    
    cv::Mat temp;
    dst.copyTo(temp);
    // Column 1D
    for(int j=0; j<src.cols; j++){
        for(int i=2; i<src.rows-2; i++){
            for(int c=0;c<3;c++){
                dst.at<cv::Vec3b>(i, j)[c] =
                (
                 1*temp.at<cv::Vec3b>(i-2, j)[c]
                 +2*temp.at<cv::Vec3b>(i-1, j)[c]
                 +4*temp.at<cv::Vec3b>(i, j)[c]
                 +2*temp.at<cv::Vec3b>(i+1, j)[c]
                 +1*temp.at<cv::Vec3b>(i+2, j)[c]
                 )/10;
            }
        }
    }
    
    return 0;
}

// Darken pixels with high saturation values.
// Convert the input image into HSV space,
// and for each pixel that has its saturation value
// larger than the `saturationThreshold`, multiply
// the value (of HSV) by the `factor`
// src - source (input) image
// dst - destination (output) image
int darkerHighSaturation(cv::Mat &src, cv::Mat &dst, int saturationThreshold, double factor){
    // Convert image to HSV representation
    cv::Mat hsv;
    cv::cvtColor(src, hsv, cv::COLOR_BGR2HSV);
    
    // Split the HSV image into separate channels
    std::vector<cv::Mat> hsvChannels;
    cv::split(hsv, hsvChannels);
    
    cv::Mat saturation = hsvChannels[1];
    cv::Mat value = hsvChannels[2];
    
    // Threshold the saturation channel that exceeds saturationThreshold
    cv::Mat saturationMask;
    cv::inRange(saturation, saturationThreshold, 255, saturationMask);

    // Reduce the value
    cv::Mat valueMasked;
    value.copyTo(valueMasked, saturationMask);
    
    valueMasked /= factor;
    valueMasked.setTo(0, valueMasked < 0);  // Clip at 0
    
    // Merge the channels back into a single image
    valueMasked.copyTo(hsvChannels[2], saturationMask);
    cv::merge(hsvChannels, hsv);

    // Convert the HSV image back to the BGR color space
    cv::cvtColor(hsv, dst, cv::COLOR_HSV2BGR);
    return 0;
}

// Save the given frames
// frame - a video frame/image
// effectFrame - a video frame/image with effects applied
int saveFrames(cv::Mat &frame, cv::Mat &effectFrame){
    std::cin.clear();
    std::string origFrameFn, effectFrameFn;
    std::cout << "Enter a filename for the original frame: ";
    std::getline(std::cin, origFrameFn);
    
    std::cin.clear();
    std::cout << "Enter a filename for the filtered frame: ";
    std::getline(std::cin, effectFrameFn);
    
    cv::imwrite(origFrameFn, frame);
    cv::imwrite(effectFrameFn, effectFrame);
    std::cout << "Done saving frames\n";
    return 0;
}

/*https:www.johndcook.com/blog/standard_deviation/*/
// A Class to calculate running statistics efficiently
RunningStat::RunningStat() : m_n(0) {}

// Clear the running statistics
void RunningStat::Clear()
{
    m_n = 0;
}

// Push a new number and calculate the new running statistics
void RunningStat::Push(double x)
{
    m_n++;

    // See Knuth TAOCP vol 2, 3rd edition, page 232
    if (m_n == 1)
    {
        m_oldM = m_newM = x;
        m_oldS = 0.0;
    }
    else
    {
        m_newM = m_oldM + (x - m_oldM)/m_n;
        m_newS = m_oldS + (x - m_oldM)*(x - m_newM);

        // set up for next iteration
        m_oldM = m_newM;
        m_oldS = m_newS;
    }
}

int RunningStat::NumDataValues() const
{
    return m_n;
}

// Retrieve the mean
double RunningStat::Mean() const
{
    return (m_n > 0) ? m_newM : 0.0;
}

// Retrieve the Variance
double RunningStat::Variance() const
{
    return ( (m_n > 1) ? m_newS/(m_n - 1) : 0.0 );
}

// Retrieve the StandardDeviation
double RunningStat::StandardDeviation() const
{
    return sqrt( Variance() );
}
