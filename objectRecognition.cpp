//
//  objectRecognition.cpp
//  Project3
//
//  2D - Object Recognition system
//  Created by Thean Cheat Lim on 2/14/23.
//

#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <cstring>
#include <map>

#include "component.hpp"
#include "csv_util.hpp"
#include "utils.hpp"

char FEATURE_DB [] = "FeatureDb.csv";
float UNKNOWN_THRESHOLD = 2;
int K = 3;

int main(int argc, char *argv[]) {
    cv::VideoCapture *capdev;

    // open the video device
    capdev = new cv::VideoCapture(0);
    if( !capdev->isOpened() ) {
            printf("Unable to open video device\n");
            return(-1);
    }

    // get some properties of the image
    cv::Size refS( (int) capdev->get(cv::CAP_PROP_FRAME_WIDTH ),
                   (int) capdev->get(cv::CAP_PROP_FRAME_HEIGHT));
    int fps = capdev->get(cv::CAP_PROP_FPS);
    printf("Expected size: %d %d\n", refS.width, refS.height);
    std::cout << "Frames per second :" << fps;

    cv::namedWindow("Video", 1); // identifies a window
    cv::Mat frame;
    cv::Mat filteredFrame;
    
    char persistKey = 'n';  // n for normal -- no effect/filter
    // For prediction use -- Read data
    std::vector<char *> trainLabels;
    std::vector<std::vector<float>> trainFeatures;
    std::vector<RunningStat>featureRunningStats;
    std::map<std::string,std::vector<std::vector<float>>> trainFeatures_byLabel;
    int fileReadStatus = -1;
    
    for(;;) {
        *capdev >> frame; // get a new frame from the camera, treat as a stream
        if( frame.empty() ) {
          printf("frame is empty\n");
          break;
        }
        
        cv::Mat labels, stats, centroids;
        std::vector<int> largestAreaLabels;
        std::map<int, std::vector<cv::Point>> component_points;
        std::vector<std::vector<cv::Point>> OOB_points;
        std::vector<std::vector<float>>OOB_stats;
        std::vector<double>OOB_angles;
        std::vector<std::string> OOB_labels;
        
        // Show Live video
        if (persistKey=='n') frame.copyTo(filteredFrame);  // No filtering Done
        if (persistKey=='t'){
            threshold(frame, filteredFrame);
        }
        if (persistKey=='m'){
            threshold(frame, filteredFrame);
            morphFilter(filteredFrame, filteredFrame);
        }
        if (persistKey=='c'){ // color-segmentation map
            threshold(frame, filteredFrame);
            morphFilter(filteredFrame, filteredFrame);
            segment(filteredFrame, filteredFrame, 1, 5, largestAreaLabels, component_points,labels, stats, centroids);
        }

        if (persistKey == 'p' or persistKey =='k') {
            // Prediction Mode
            // p -- nearest neighbor
            // k -- multiclass k nearest neighbor
            threshold(frame, filteredFrame);
            morphFilter(filteredFrame, filteredFrame);
            segment(filteredFrame, filteredFrame, 0, 5, largestAreaLabels, component_points, labels, stats, centroids);
            OOBFeatures(component_points, OOB_points, OOB_stats, OOB_angles);
            frame.copyTo(filteredFrame);
            
            // Compute feature for each component in the input frame
            std::vector<std::vector<float>> testFeatures;
            featurize(OOB_stats, labels, largestAreaLabels, testFeatures);
            
            if (trainFeatures.empty()){
                // Read data
                fileReadStatus = read_image_data_csv(FEATURE_DB, trainLabels, trainFeatures,0);
                if (fileReadStatus == -1){
                    OOB_labels.clear();
                    // Naming OOB labels
                    for (int i = 0; i<OOB_points.size(); i++){
                        OOB_labels.push_back("Unknown");
                        std::cout<< i <<std::endl;
                    }
                    frame.copyTo(filteredFrame);
                    drawOOB(filteredFrame, OOB_points, OOB_stats, OOB_angles, OOB_labels);
                } else {
                    // Collect useful info
                    int trainDataCnt = (int)trainFeatures.size();
                    int featureCnt = (int)trainFeatures[0].size();
                    
                    // Compute Running Statistic for each column/feature
                    // Initialize featureRunningStats
                    for(int f=0; f<featureCnt; f++)featureRunningStats.push_back(RunningStat());
                    
                    // Also separate data by it labels/classes
                    for (int i = 0; i<trainDataCnt; i++){
                        // Separate data by it labels/classes
                        std::string label = trainLabels[i];
                        if (not trainFeatures_byLabel.contains(label)) {
                            std::vector<std::vector<float>> temp;
                            trainFeatures_byLabel[label] = temp;
                        }
                        trainFeatures_byLabel[label].push_back(trainFeatures[i]);
                        // Calculate Running Stats
                        for (int f = 0; f<featureCnt; f++){
                            featureRunningStats[f].Push(trainFeatures[i][f]);
                        }
                    }
                }
            }
            // If able to read database file
            if (fileReadStatus == 0){
                // Collect useful info
                int testDataCnt = (int) testFeatures.size();
                int trainDataCnt = (int)trainFeatures.size();
                int featureCnt = (int)trainFeatures[0].size();
                
                // Vector of predictions and distances
                std::vector<std::string> predictedLabels;
                std::vector<std::pair<float, std::string>> distances;
                if (persistKey == 'p'){
                    // Compute distance for each component in the input frame
                    for (int test = 0; test<testDataCnt; test++){
                        // For each inputFrame component, compute its distance to trainFeatures
                        for(int train = 0; train < trainDataCnt; train++) {
                            // Normalized (by StdDev) Euclidean Distance
                            float distance = 0.0f;
                            // For each feature
                            for(int f = 0; f<featureCnt; f++){
                                float delta = (trainFeatures[train][f]-testFeatures[test][f])/(featureRunningStats[f].StandardDeviation()+0.000000001);
                                distance+=(delta * delta);
                            }
                            distance/=featureCnt;
                            std::pair<float, std::string> distPair(distance, trainLabels[train]);
                            distances.push_back(distPair);
                        }
                        // Look for the smallest distance
                        // Sort ascending
                        std::sort(distances.begin(), distances.end());
                        // Push into prediction vector
                        if (distances[0].first > UNKNOWN_THRESHOLD){
                            predictedLabels.push_back("Unknown");
                        } else {
                            predictedLabels.push_back(distances[0].second);
                        }
                        distances.clear();
                    }
                }
                if (persistKey == 'k'){
                    // Compute distance for each component in the input frame
                    for (int test = 0; test<testDataCnt; test++){
                        // For each class/ label
                        for (auto const& [label, trainFeatureVec_fixedLabel] : trainFeatures_byLabel){
                            // Normalized (by StdDev) Euclidean Distance
                            std::vector<float> distances_fixedLabel;
                            int trainCnt_fixedLabel = (int)trainFeatureVec_fixedLabel.size();
                            // For each trainFeature within the same class/label
                            for (int train = 0; train<trainCnt_fixedLabel; train++){
                                float distance = 0.0f;
                                // For each feature
                                for(int f = 0; f<featureCnt; f++){
                                    float delta = (trainFeatureVec_fixedLabel[train][f]-testFeatures[test][f])/(featureRunningStats[f].StandardDeviation()+0.000000001);
                                    distance+=(delta * delta);
                                }
                                distance/=featureCnt;
                                distances_fixedLabel.push_back(distance);
                            }
                            // Sort ascending
                            std::sort(distances_fixedLabel.begin(), distances_fixedLabel.end());
                            float topKTotalDist_fixedLabel = 0.0f;
                            for (int k = 0; k<K ;k++) topKTotalDist_fixedLabel+=distances_fixedLabel[k];
                            // Push to overall distances across classes/labels
                            std::pair<float, std::string> distPair(topKTotalDist_fixedLabel, label);
                            distances.push_back(distPair);
                        }
                        // Look for the smallest distance across classes/labels
                        // Sort ascending
                        std::sort(distances.begin(), distances.end());
                        // Push into prediction vector
                        if (distances[0].first > UNKNOWN_THRESHOLD){
                            predictedLabels.push_back("Unknown");
                        } else {
                            predictedLabels.push_back(distances[0].second);
                        }
                        distances.clear();
                    }
                }
                // Draw the oriented bounding box with predicted labels on it
                drawOOB(filteredFrame, OOB_points, OOB_stats, OOB_angles, predictedLabels);
                predictedLabels.clear();
            }
        }
        
        cv::imshow("Video", filteredFrame);
        
        // see if there is a waiting keystroke
        char key = cv::waitKey(10);
        
        if( key == 'q') break;
        if (key == 's') {
            saveFrames(frame, filteredFrame);
            key = persistKey;
        }
        if (key == 'a') {
            OOB_labels.clear();
            // Naming OOB labels
            for (int i = 0; i<OOB_points.size(); i++){
                OOB_labels.push_back(std::to_string(i));
                std::cout<< i <<std::endl;
            }
            frame.copyTo(filteredFrame);
            drawOOB(filteredFrame, OOB_points, OOB_stats, OOB_angles, OOB_labels);
            cv::imshow("Adding to Database", filteredFrame);
            cv::waitKey(10);
            
            std::cin.clear();
            // append/add to db
            std::vector<std::vector<float>> featureVec;
            featurize(OOB_stats, labels, largestAreaLabels, featureVec);

            int toLabel = 1; // want to label
            while(toLabel){
                std::string labelStr;
                int labelNum;
                
                if (component_points.size()) std::cout << "Choose a label between 0 and "<<component_points.size()-1<<" : \n";
                std::cin >> labelNum;
                if (labelNum<0 or labelNum>=component_points.size()){
                    std::cout<<"Incorrect label number chosen. Please try again."<<std::endl;
                    continue;
                }
                std::cout << "Enter a label: ";
                std::cin.ignore();
                std::getline(std::cin, labelStr);
                
                append_image_data_csv(FEATURE_DB, &labelStr[0], featureVec[labelNum], 0);
                std::cout<<"Done Saving Label and features. Press key l to resume."<<std::endl;
                std::cout<<"Do you want to keep labeling? Press 1 for yes and 0 for no. \n**To use KNN classification, you need "<<K<<" training images per object.**"<<std::endl;
                std::cin >> toLabel;
            }
            std::cout<<"Done attaching labels to images. Please return to the live video."<<std::endl;
            // Clean out Prediction cache
            trainLabels.clear();
            trainFeatures.clear();
            featureRunningStats.clear();
            trainFeatures_byLabel.clear();
            
            // back to previous key
            key = persistKey;
            // Destroy current window that's used to add label to data
            cv::destroyWindow("Adding to Database");
        }

        // Persist key
        if (strchr("ntmcpk", key)){ /*https:stackoverflow.com/a/19548575/19481647*/
            persistKey = key;
        }
    }

    delete capdev;
    return(0);
}
