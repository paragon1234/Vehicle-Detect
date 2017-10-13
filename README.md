**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* We have also applied a color transform and append binned color features, as well as histograms of color, to HOG feature vector. 
* For the above steps, normalize the features and randomize a selection for training and testing.
* Implement a sliding-window technique and use the trained classifier to search for vehicles in images.
* Run the pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./examples/car.png
[image2]: ./examples/notCar.png
[image3]: ./examples/searchGrid0.jpg
[image4]: ./examples/searchGrid1.jpg
[image5]: ./examples/searchGrid2.jpg
[image6]: ./examples/searchGrid3.jpg
[image7]: ./examples/test1.jpg
[image8]: ./examples/heatLabel.jpg
[image9]: ./examples/final.jpg
[image10]: ./examples/HOG.jpg
[video1]: ./project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in lines 239-250 VehicleDetect.py.  

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]          

![alt text][image2]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example of HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:


![alt text][image10]

#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and settled on the final value based upon the performance of the SVM classifier produced using them (training accuracy of the classifier). My final parameters were YUV colorspace, 9 orientations, 16 pixels per cell, 2 cells per block, and ALL channels of the colorspace. Also, 32 bins for histogram and 16 bins for color.

For colorspace all the colorspcase except RGB performed well. When using all the channels, training accuracy was higher but it required substantial computation time. Using only Y channel of LUV color-scpae, there was a reduction of only 0.5% in the training accuracy but  redreduced computation time by half. Hence, I preferred it as the output quality does'nt change much. 32 bins Histogram, 16 bin colors and 9 orientation gave the training accuracy of over 99%. 8 pixels per cell and 16 pixels per cell both gave training accuracy of over 99%. 16 was preferred because of low computation time.

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM with the parameter C=0.01 and using HOG features, channel intensity and histogram features. Using this combination I was able to achieve a test accuracy of 98.68% (using all channels of YUV colorsace gave training accuracy of 99.17%).

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I have used a more efficient method for doing the sliding window approach, one that allows us to only have to extract the Hog features once. The code below defines a single function find_cars that's able to both extract features and make predictions.

The find_cars only has to extract hog features once and then can be sub-sampled to get all of its overlaying windows. Each window is defined by a scaling factor where a scale of 1 would result in a window that's 8 x 8 cells then the overlap of each window is in terms of the cell distance. This means that a cells_per_step = 2 would result in a search window overlap of 75%. Its possible to run this same function multiple times for different scale values to generate multiple-scaled search windows.

I decided to search window positions at the center of the image with small scales and in the lower half of the image using medium and large scale.

There were 5 cases that I simulated on test and project video:
* Medium and high scale with no heatThresholding and no data from pevious frame
* Medium and high scale with no heatThresholding and data from previous frame
* Medium and high sclae with heatThresholding and data from previous frame
* Low, medium and high scale with no thresholding and no data from previoud frame
* Low, medium and high scale with thresholding and data from previous frame

Initially I used a small scale of 0.5. However,  Small scale did not detect small vehicles and provided false positive. Hence, it is not required. I then changed my lowest scale to 0.75. This worked as the vehicles were detected reliably (using 0.5 scale, there were certain frame when no vehicle was detected). Medium and high scale were essential as they provide true positive with very few false positive. 

For certain cases only a single heat map is obtained for the car on medium/high scale with 75% overlap. Hence, whether to use heatThresholding was a major decision point. I preferred to use heatThresholding only when I have data from previous frame or a scale of 0.5/0.75 is used. 
    
It has been found that the lower scales work best when the overlapping of sliding windows is as high as 87.5% or more. However, for higher scales the overlapping can be lower. Hence, I made overalpping of sliding windows dependent on scale. For small scale the windows are close together and for large scale the windows are far apart(line 134-136 in file VehicleDetect.py)


Based on the result of the simulation of the above 5 cases, I peferred the case of "Medium and high scale with heatThresholding and data from previous frame" (line 309-337 in file VehicleDetect.py). I used a simple add operation to combine the heatmap from current frame and previous frame (line 323-325 in file VehicleDetect.py).

My four search sliding window, each with a distinct scale are (line 315-318 in file VehicleDetect.py):

![alt text][image3]

![alt text][image4]

![alt text][image5]

![alt text][image6]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on four scales using L-channel of LUV for HOG features, plus spatially binned color and histograms of color in the feature vector (functions extract_features line 70 and function find_cars line 111 in VehicleDetect.py), which provided a nice result.  Here is an example image:

![alt text][image7]
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap  to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected. 

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

##### Here is the heatmaps  and output of `scipy.ndimage.measurements.label()` on the integrated heatmap

![alt text][image8]


##### Here the resulting bounding boxes are shown:
![alt text][image9]



---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Most of the problem I faced were related to detection accuracy and also about the speed of execution. How to use data from previous frame was also a design consideration. Another consideration is that in a lot of places cars are detected only at one scale and that too in a single window (not in its overlapped window). In such cases not using heatThresholding gives better result. As discussed in next paragraph it is better not to use small scale, without which we can discard heatThresholding.

The pipeline is probably most likely to fail in cases where vehicles don't resemble those in the training dataset. Lighting and environmental conditions might also play a role (e.g. a white car against a white background). Distant cars are an issue(small window scales produce more false positives, but they also did not often correctly label the smaller, distant cars). Moreover, the lowest scale took the maximum time for the detection of the vehicle (on removing the lowest scale the time on test_video reduced from 280s to 80s). As such, it is better not to use small scale (as they do not detect small cars but detect a lot of false positive, and its computing consuming a lot of time). Not considering them will help to discarding heatThresholding.

Oncomming cars are also an issue. This may be due to the fact that training data contain images of vehicles from behind. As a result it is difficult to detect vehicles from front. This would be problem while driving on narrow lanes (say 2 lanes without the divider). This can be resolved by augmenting the training set by incorporating front images of vehicle. However, oncomming cars are least suitable for considering data from pervious frame. So, in case we use them with data from pevious frame, then some better algorithm (like searching again in those regions) will be required.

Things could be made better by taking the estimated coordinates of the trapezoid region of road in the videos and using scale factors to match them suitably. In this way, we can discard false positives by not considering the points lying outside the trapezoid region of road. However, these will work for specific lane and will not be generalizable. Also, these may give problem when the lane has sharper curve or is more broad. In such cases, the lane detection algorithm of second order polynomial can be used which detect the lane lines and also use those co-ordinates of the lane to look out for vehicles.

## Changes
My first attempt resulted in the [output video](./project_video_outputi_6.mp4). At certain places, no vehicle was detected and there were a lot of false positive. To fix those issues, I made following changes:
1) Changed 'C' parameter of LinearSVC to 0.01, to reduce errors.
2) It was observed that at certain locations the vehicles were not detected. To fix this I made 2 chanes:
    * Changed the smallest scale from 0.5 to 0.75. This worked and the vehicles were detected reliably. 
    * I have used data from previous frames, thresholding of 8 with dequeue depth of 8. 
3) The above step results in increased computation time. To speed up computation:
    * I used only L channel of LUV color-space, for HOG Computation. This result in slight reduction of training accuracy by 0.5%, but provided substantial reduction in computation time with no impact on the output.
    * It has been found that the lower scales work best when the overlapping of sliding windows is as high as 87.5% or more. However, for higher scales the overlapping can be lower. Hence, I made overalpping of sliding windows dependent on scale. For small scale the windows are close together and for large scale the windows are far apart(line 134-136 in file VehicleDetect.py)
    * On using lower scale of 0.75, it was observed that there was no impact on output if I drop the scale of 2.5. Hence, scale of 2.5 not used
4) 35 Images of non-vehicles were used to supress false positives. However, there are still false positives and they cannot be removed completely. I think that we need to generate larger dataset of non-vehicles to remove those false positives.
