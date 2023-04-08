# RealTime2DObjectRecognition

## Description
I developed a real-time 2D object recognition system. The first step involves thresholding the colored image (or video frame) to convert it to a binary image, distinguishing between the foreground (the object) and the background. Next, morphological filters are applied to clean up the object. Image segmentation is then used to identify the object region, and the features of the object are calculated. The features were then compared to those extracted from training data, and the input image was classified based on its (K) closest neighbors.

## Demo
### Threshold the input video
<img src="/images/Threshold.png" width="300" height="200">

### Segment the image into regions
<img src="/images/MultipleSegmentation.png" width="300" height="200">

### Oriented Bounding Box
<img src="/images/OrientedBoundingBoxes.png" width="300" height="200">

  To achieve this, I first calculated the angle to the axis of the least central moment. Then, I rotated and projected each component's coordinates in the reverse direction. Next, I identified the four corners of the oriented bounding box and rotated them back to their original space.  
  Additionally, I added a number to each bounding box, starting from 0. These numbers help users identify which object or box they want to label or classify, particularly when multiple objects are present. During the prediction mode, the predicted class/label for each box is placed on top of the box. The text is scaled proportionally to the size of the bounding box.   
  Note that the texts are oriented similarly to the bounding box using the same rotation approach mentioned above. OpenCV only supports adding texts horizontally, which can result in texts getting cut off when they are at the edge of an image. However, the same text could fit into the frame when it is rotated. Therefore, extra work was done to take care of this edge case.  

### Multi-object Classification Using KNN
<img src="/images/Classification.jpeg" width="300" height="200">   

  The Mean Scaled Euclidean distance metric was employed to measure the similarity between objects. To distinguish between known and unknown objects, a threshold distance value of 2 was chosen. Objects with a distance greater than 2 were classified as unknown. The results indicate that all predictions were accurate, which may be attributed to the distinct shapes of each object.  
  
  Features used were: the height-to-width ratio, the percentage of the oriented bounding box filled, and the first Hu moment, which is also known as the centroid moment. The first Hu moment represents the ratio of central x and y moments to its area. It measures how spread out the object is around its centroid, with smaller values indicating a more compact shape and larger values indicating a more spread-out shape.

### Real Time Demo
1. Multiple Objects Recognition: https://drive.google.com/file/d/1sHB-tjR9BK_2al4UmgTQfZLogU5vbwdO/view 
2. Recognizing Unknown Objects: https://drive.google.com/file/d/1P6Vl8hKe6ir3X5t8BY_z2pMxyPMqcpWg/view  

## Instructions
Run `objectRecognition.cpp`
Some useful hotkeys:
- s = Save Frame
- q = Quit program
- p = Nearest Neighbor classification
- k = 3-NN classification
- a = press at any time after pressing k or p so that you can attach a label to a bonding box.
- t = thresholded binary image
- m = threshold + morphological filtered binary image
- c = threshold + morphological filtered binary image. Segmented and colored the top 5 largest regions in the image.

**More notes**:
  To gather training data, users can activate the prediction mode by pressing the 'p' key for the nearest-neighbor classifier or the 'k' key for the 3-NN classifier. The live video input is dynamically thresholded, cleaned, and segmented, and the top 5 largest items are identified and marked with oriented bounding boxes. The predicted labels for each item are displayed alongside their respective bounding boxes. If an object is not recognized, it is labeled as an "Unknown".  

  Users can press' a' to add a new label to the database, and the bounding boxes are sequentially numbered starting from 0. They are then prompted to enter the bounding box number and the corresponding label or class. This approach can also be used when the system made the wrong predictions – users can add more data to navigate the system to the correct future predictions.  
  
## OS and IDE
OS:
MacOS Ventura 13.0.1 (22A400)

IDE:
XCode


## Acknowledgement
- Find the orientation of an image: https://stackoverflow.com/questions/11330269/find-the-orientation-of-an-image?answertab=active#tab-top 
- Rotating Points in Two-Dimensions: https://danceswithcode.net/engineeringnotes/rotations_in_2d/rotations_in_2d.html 
- Tutorial Eroding and Dilating: https://docs.opencv.org/3.4/db/df6/tutorial_erosion_dilatation.html 
- Check if element is in the list (contains): https://stackoverflow.com/questions/24139428/check-if-element-is-in-the-list-contains 
- Accurately computing running variance: https://www.johndcook.com/blog/standard_deviation/ 
- Comparing a char to chars in a list in C++: https://stackoverflow.com/questions/19548439/comparing-a-char-to-chars-in-a-list-in-c/19548575#19548575 
- What is the best way to sort a vector leaving the original one unaltered?: https://stackoverflow.com/questions/47537049/what-is-the-best-way-to-sort-a-vector-leaving-the-original-one-unaltered 
- Opencv tutorials/documentation in general
