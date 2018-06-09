# Vehicle Detection Project

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

I tried various combinations of parameters and settled on the final value based upon the performance of the SVM classifier produced using them (training accuracy of the classifier). My final parameters were LUV colorspace, 9 orientations, 8 pixels per cell, 2 cells per block, and L channel of the LUV colorspace for HOG computation. Also, 32 bins for histogram and 16 bins for color.

For colorspace all the colorspcase except RGB performed well. When using all the channels, training accuracy was higher but it required substantial computation time. Using only L channel of LUV color-scpae, there was a reduction of only 0.5% in the training accuracy but  reduced computation time by half. Hence, I preferred it as the output quality does'nt change much. 32 bins Histogram, 16 bin colors and 9 orientation gave the training accuracy of over 98.67%. 8 pixels per cell and 16 pixels per cell both gave training accuracy of over 98%. 8 was preferred because it provided better output video.

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM with the parameter C=0.01 and using HOG features, channel intensity and histogram features. Using this combination I was able to achieve a test accuracy of 98.68% (using all channels of LUV colorsace gave training accuracy of 99.17%).

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

Based on the result of the simulation of the above 5 cases, I peferred the case of "Medium and high scale with heatThresholding and data from previous frame" (line 309-337 in file VehicleDetect.py). I used a simple add operation to combine the heatmap from current frame and previous 14 frame (line 323-325 in file VehicleDetect.py).

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

[40 Images](./nonVehImageAdditional) of non-vehicles were used to supress false positives. However, there are still false positives and they cannot be removed completely. I think that we need to generate larger dataset of non-vehicles to remove those false positives. Another reason could be that we have more data for non-Vehicle than for vehicles. Probably, the dataset should be more balanced. However, the false detection are outside the road. This is less worrisome and could be eliminated by using lane detection, where the detection outside the road lanes could be truncated.

Things could be made better by taking the estimated coordinates of the trapezoid region of road in the videos and using scale factors to match them suitably. In this way, we can discard false positives by not considering the points lying outside the trapezoid region of road. However, these will work for specific lane and will not be generalizable. Also, these may give problem when the lane has sharper curve or is more broad. In such cases, the lane detection algorithm of second order polynomial can be used which detect the lane lines and also use those co-ordinates of the lane to look out for vehicles.

## Changes
My first attempt resulted in the [output video](./project_video_outputi_6.mp4). At certain places, no vehicle was detected and there were a lot of false positive. To fix those issues, I made following changes:
1) Changed 'C' parameter of LinearSVC to 0.01, to reduce errors.
2) It was observed that at certain locations the vehicles were not detected. To fix this I made 2 chanes:
    * Changed the smallest scale from 0.5 to 0.75. This worked and the vehicles were detected reliably. 
    * I have used data from previous frames, thresholding of 15 with dequeue depth of 15. 
3) The above step results in increased computation time. To speed up computation:
    * I used only L channel of LUV color-space, for HOG Computation. This result in slight reduction of training accuracy by 0.5%, but provided substantial reduction in computation time with no impact on the output.
    * (TODO: THIS IS NOT DONE. It has been found that the lower scales work best when the overlapping of sliding windows is as high as 87.5% or more. However, for higher scales the overlapping can be lower. Hence, I made overalpping of sliding windows dependent on scale. For small scale the windows are close together and for large scale the windows are far apart(line 134-136 in file VehicleDetect.py))
    * On using lower scale of 0.75, it was observed that there was no impact on output if I drop the scale of 2.5. Hence, scale of 2.5 not used
4) [40 Images](./nonVehImageAdditional) of non-vehicles were used to supress false positives. However, there are still false positives and they cannot be removed completely. I think that we need to generate larger dataset of non-vehicles to remove those false positives. Another reason could be that we have more data for non-Vehicle than for vehicles. Probably, the dataset should be more balanced. However, the false detection are outside the road. This is less worrisome and could be eliminated by using lane detection, where the detection outside the road lanes could be truncated.

## Yolo
### Description
In this project we will implement tiny-YOLO v1. Full details of the network, training and implementation are available in the paper - http://arxiv.org/abs/1506.02640

YOLO divides the input image into an SxS grid. If the center of an object falls into a grid cell, that grid cell
is responsible for detecting that object. Each grid cell predicts B bounding boxes and confidence scores for those boxes.

Confidence is defined as (Probability that the grid cell contains an object) multiplied by (Intersection over union of predicted bounding box over the ground truth). Or

    Confidence = Pr(Object) x IOU_truth_pred.                                                      (1)

Each bounding box consists of 5 predictions:
1. x
2. y
3. w
4. h
5. confidence

The (x; y) coordinates represent the center of the box relative to the bounds of the grid cell. The width
and height are predicted relative to the whole image. Finally the confidence prediction represents the IOU between the
predicted box and any ground truth box.

Each grid cell also predicts C conditional class probabilities, Pr(ClassijObject). These probabilities are conditioned
on the grid cell containing an object. We only predict one set of class probabilities per grid cell, regardless of the
number of boxes B.

At test time we multiply the conditional class probabilities and the individual box confidence predictions,

    Pr(Class|Object) x Pr(Object) x IOU_truth_pred = Pr(Class) x IOU_truth_pred                    (2)

which gives us class-specific confidence scores for each box. These scores encode both the probability of that class appearing in the box and how well the predicted box fits the object.

So at test time, the final output vector for each image is a **S x S x (B x 5 + C)** length vector

### Output
The output is a feature vector of dimension S x S x (B x 5 + C). How can we be sure that the output contains B bounding boxes per grid with 5 parameters. The output can be anything. However while training the network, you will feed it with 2 things :

    X - image
    y - pre-processed feature vector of dimensions (S x S x (B x 5 + C))

As you feed your model with more such (X, y) pairs it will begin to learn the correlation between the objects, their bounding boxes and the feature vectors (y). That is the magic of deep neural networks.

### Post-Processing
 
The model was trained on PASCAL VOC dataset. We use S = 7, B = 2. PASCAL VOC has 20 labelled classes so C = 20. So our final prediction, for each input image, is:

    output tensor length = S x S x (B x 5 + C)
    output tensor length = 7 x 7 x (2x5 + 20)
    output tensor length = 1470.

The structure of the 1470 length tensor is as follows:

1. First 980 values correspons to probabilities for each of the 20 classes for each grid cell. These probabilities are conditioned on objects being present in each grid cell.
2. The next 98 values are confidence scores for 2 bounding boxes predicted by each grid cells.
3. The next 392 values are co-ordinates (x, y, w, h) for 2 bounding boxes per grid cell.

As you can see in the above image, each input image is divided into an S x S grid and for each grid cell, our model predicts B bounding boxes and C confidence scores. There is a fair amount of post-processing involved to arrive at the final bounding boxes based on the model's predictions.

#### Class score threshold
We reject output from grid cells below a certain threshold (0.2) of class scores (equation 2), computed at test time.

#### Reject overlapping (duplicate) bounding boxes
If multiple bounding boxes, for each class overlap and have an IOU of more than 0.4 (intersecting area is 40% of union area of boxes), then we keep the box with the highest class score and reject the other box(es).

#### Drawing the bounding boxes
The predictions (x, y) for each bounding box are relative to the bounds of the grid cell and (w, h) are relative to the whole image. To compute the final bounding box coodinates we have to multiply `w` & `h` with the width & height of the portion of the image used as input for the network.

