# Vehicle Detection And Tracking
A combination of advanced computer vision and machine learning techniques are used to create a vehicle detection pipeline, which is used to detect and track vehicles in a video from a front facing camera in a car.

The following steps were taken in coming up with the pipeline:
1. Explore color spaces to come up with a color space that can be used to distinguish between vehicles and non-vehicle images from a standard dataset.
2. Explorethe histogram of oriented gradients (HOG) features and come up with a set of parameters to reliably distinguish vehicles from non-vehicles
3. Extract a combination of spatial features, color histograms, and HOG features that can be used to train a classifier using standard machine learning techniques liks decisions trees, support vector machine (SVM) or even neural networks
4. An SVM with a non-linear RBF kernel is used to train a vehicle  classifer to detect vehicles with a high degree of accuracy (> 99% on the test set)
5. Used a sliding window approach, along with HOG sub-sampling for efficiency, to come up with detect a vehicle in a test camera image
6. Tuned windows at a mutliple scales to detect vehicle at various distances from the camera
7. Handled multiple detections and false positives by creating a heat map and placing the vehicle in the area beyond a certain heat threshold
8. Finally put all this together to create a pipeline that was run on test images, the test video, and the project video

The detailed steps along with example images is in the VehicleDetecionAndTracking.ipynb Jupyter notebook

# Running the code

## Anaconda and Jupyter Notebook

`jupyter notebook VehicleDetecionAndTracking.ipynb`


## Python 3
Some of the code within the jupyter notebook has been pulled out into standalone python modules

ExploreColorSpace.py
TrainClassifier.py
VehicleDetector.py


# Packages needed for Python3

* OpenCv
* Numpy
* MatplotLib
* MoviePy
* sklearn
* pickle
* scipy
* skimage

# Output

## Test video
Test video output with vehicle detection and tracking is in video_output folder: test_video_out.mp4

## Project video
Project video output with vehicle detection and tracking is in video_output folder: project_video_out_final.mp4

## Output images
Added various output images to the output_images folder
