'''
Created on Dec 11, 2017

@author: kiniap
'''

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import cv2
import glob
import time
import random
import pickle
from sklearn.svm import LinearSVC, SVC
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
from scipy.ndimage.measurements import label
from collections import deque
# NOTE: the next import is only valid for scikit-learn version <= 0.17
#from sklearn.cross_validation import train_test_split
# for scikit-learn >= 0.18 use:
from sklearn.model_selection import train_test_split
# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip
from IPython.display import HTML

# Define a function to compute binned color features  
def bin_spatial(img, size=(32, 32)):
    # Use cv2.resize().ravel() to create the feature vector
    features = cv2.resize(img, size).ravel() 
    # Return the feature vector
    return features

# Define a function to compute color histogram features  
def color_hist(img, nbins=32, bins_range=(0, 256)):
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:,:,0], bins=nbins, range=bins_range)
    channel2_hist = np.histogram(img[:,:,1], bins=nbins, range=bins_range)
    channel3_hist = np.histogram(img[:,:,2], bins=nbins, range=bins_range)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return hist_features

# Define a function to return HOG features and visualization
def get_hog_features(img, orient, pix_per_cell, cell_per_block, 
                        vis=False, feature_vec=True):
    # Call with two outputs if vis==True
    if vis == True:
        features, hog_image = hog(img, orientations=orient, 
                                  pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block), 
                                  transform_sqrt=False, 
                                  visualise=vis, feature_vector=feature_vec, block_norm='L2-Hys')
        return features, hog_image
    # Otherwise call with one output
    else:      
        features = hog(img, orientations=orient, 
                       pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block), 
                       transform_sqrt=False, 
                       visualise=vis, feature_vector=feature_vec, block_norm='L2-Hys')
        return features


# Define a single function that can extract features using hog sub-sampling and make predictions
def find_cars_in_image(img, color_space, hog_channel, ystart, ystop, scale, svc, X_scaler, 
                       orient, pix_per_cell, cell_per_block, spatial_size, hist_bins, cells_per_step = 2, get_all_boxes=False):
    
    #draw_img = np.copy(img)
    img = img.astype(np.float32)/255
    img_tosearch = img[ystart:ystop,:,:]
    
    #Apply color conversion if other than 'RGB'
    if color_space != 'RGB':
        if color_space == 'HSV':
            ctrans_tosearch = cv2.cvtColor(img_tosearch, cv2.COLOR_RGB2HSV)
        elif color_space == 'LUV':
            ctrans_tosearch = cv2.cvtColor(img_tosearch, cv2.COLOR_RGB2LUV)
        elif color_space == 'HLS':
            ctrans_tosearch = cv2.cvtColor(img_tosearch, cv2.COLOR_RGB2HLS)
        elif color_space == 'YUV':
            ctrans_tosearch = cv2.cvtColor(img_tosearch, cv2.COLOR_RGB2YUV)
        elif color_space == 'YCrCb':
            ctrans_tosearch = cv2.cvtColor(img_tosearch, cv2.COLOR_RGB2YCrCb)
    else: ctrans_tosearch = np.copy(img)      
 
    if scale != 1:
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))

    # Define blocks and steps as above
    nxblocks = (ctrans_tosearch[:,:,0].shape[1] // pix_per_cell) - cell_per_block + 1
    nyblocks = (ctrans_tosearch[:,:,0].shape[0] // pix_per_cell) - cell_per_block + 1 
    nfeat_per_block = orient*cell_per_block**2
    
    # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
    window = 64
    nblocks_per_window = (window // pix_per_cell) - cell_per_block + 1
    #cells_per_step = 2  # Instead of overlap, define how many cells to step
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step
    
    # Compute individual channel HOG features for the entire image    
    if hog_channel == "ALL":
        hog1 = get_hog_features(ctrans_tosearch[:,:,0], orient, pix_per_cell, cell_per_block, feature_vec=False)
        hog2 = get_hog_features(ctrans_tosearch[:,:,1], orient, pix_per_cell, cell_per_block, feature_vec=False)
        hog3 = get_hog_features(ctrans_tosearch[:,:,2], orient, pix_per_cell, cell_per_block, feature_vec=False)
    else:
        hog1 = get_hog_features(ctrans_tosearch[:,:,hog_channel], orient, pix_per_cell, cell_per_block, feature_vec=False)
    
    # initialize array of boxes to return all the boxes where cars are detected
    boxes = []
    for xb in range(nxsteps):
        for yb in range(nysteps):
            ypos = yb*cells_per_step
            xpos = xb*cells_per_step
            # Extract HOG for this patch
            if hog_channel == "ALL":
                hog_feat1 = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
                hog_feat2 = hog2[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
                hog_feat3 = hog3[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
                hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))
            else:
                hog_features = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
                
            xleft = xpos*pix_per_cell
            ytop = ypos*pix_per_cell

            # Extract the image patch
            subimg = cv2.resize(ctrans_tosearch[ytop:ytop+window, xleft:xleft+window], (64,64))
          
            # Get color features
            spatial_features = bin_spatial(subimg, size=spatial_size)
            hist_features = color_hist(subimg, nbins=hist_bins)

            # Scale features and make a prediction
            test_features = X_scaler.transform(np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1))    
            #test_features = X_scaler.transform(np.hstack((shape_feat, hist_feat)).reshape(1, -1))    
            test_prediction = svc.predict(test_features)
            
            #test_features = X_scaler.transform(np.hstack(hog_features).reshape(1, -1))    
            #test_prediction = svc.predict(test_features)
            
            if (test_prediction == 1) or get_all_boxes:
                xbox_left = np.int(xleft*scale)
                ytop_draw = np.int(ytop*scale)
                win_draw = np.int(window*scale)
                #cv2.rectangle(draw_img,(xbox_left, ytop_draw+ystart),(xbox_left+win_draw,ytop_draw+win_draw+ystart),(0,0,255),6) 
                boxes.append(((xbox_left, ytop_draw+ystart),(xbox_left+win_draw,ytop_draw+win_draw+ystart)))

    return boxes

# Define a function to draw bounding boxes
def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    # Make a copy of the image
    imcopy = np.copy(img)
    # Iterate through the bounding boxes
    for bbox in bboxes:
        # Draw a rectangle given bbox coordinates
        cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
    # Return the image copy with boxes drawn
    return imcopy

def add_heat(heatmap, bbox_list):
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    # Return updated heatmap
    return heatmap# Iterate through list of bboxes

def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return heatmap

def draw_labeled_bboxes(img, labels):
    # Iterate through all detected cars
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 6)
    # Return the image
    return img

class VehicleDetector():
    def __init__(self):
        self.heatmaps = deque(maxlen=10)
    

    def vehicle_detection_pipeline(self, image):
        # initialize current boxes to empty list
        current_boxes=[]
    
        # Scale 1.1
        ystart1_1 = 380
        ystop1_1 = 560
        scale1_1 = 1.1
        boxes1_1 = find_cars_in_image(image, color_space, hog_channel, ystart1_1, ystop1_1, scale1_1, svc, X_scaler, 
                            orient, pix_per_cell, cell_per_block, spatial_size, hist_bins, get_all_boxes=False)
        current_boxes.extend(boxes1_1)

        # Scale 1.5
        ystart1_5 = 400
        ystop1_5 = 650
        scale1_5 = 1.5
        boxes1_5 = find_cars_in_image(image, color_space, hog_channel, ystart1_5, ystop1_5, scale1_5, svc, X_scaler, 
                           orient, pix_per_cell, cell_per_block, spatial_size, hist_bins, get_all_boxes=False)
        current_boxes.extend(boxes1_5)

        # Scale 2.0
        ystart2 = 400
        ystop2 = 650
        scale2 = 1.9
        boxes2 = find_cars_in_image(image, color_space, hog_channel, ystart2, ystop2, scale2, svc, X_scaler, 
                           orient, pix_per_cell, cell_per_block, spatial_size, hist_bins, get_all_boxes=False)
        current_boxes.extend(boxes2)

        # Scale 3.0
        ystart3 = 400
        ystop3 = 700
        scale3 = 2.8
        boxes3 = find_cars_in_image(image, color_space, hog_channel, ystart3, ystop3, scale3, svc, X_scaler, 
                           orient, pix_per_cell, cell_per_block, spatial_size, hist_bins, cells_per_step = 1, get_all_boxes=False)
        current_boxes.extend(boxes3)
    
        heat = np.zeros_like(image[:,:,0]).astype(np.float)
        # Add heat to each box in box list
        heat = add_heat(heat,current_boxes)
    
        # Visualize the heatmap when displaying    
        heat = np.clip(heat, 0, 255)
        # Store the current heatmaps into the list of heatmaps
        self.heatmaps.append(heat)
        # Compute the sum of store heatmaps
        heatmap_sum = sum(self.heatmaps)
        
        # Apply threshold to help remove false positives
        average_heatmap = apply_threshold(heatmap_sum,1+len(self.heatmaps)//2)
        
        # Apply labels to the heatmap using label() function from scipy.ndimage.measurements
        labels = label(average_heatmap)
        number_of_cars_found = labels[1]
    
        # Draw labeled boxes
        draw_img = draw_labeled_bboxes(np.copy(image), labels)
    
        return draw_img
    
dist_pickle = pickle.load( open('svm_model/svm_rbf_model_parameters_spatial_hist_20_YUV_orient_9_pix_per_cell_16_cell_per_block_2.p', "rb" ) )
svc = dist_pickle["svc"]
color_space = dist_pickle["color_space"]
X_scaler = dist_pickle["scaler"]
orient = dist_pickle["orient"]
pix_per_cell = dist_pickle["pix_per_cell"]
cell_per_block = dist_pickle["cell_per_block"]
spatial_size = dist_pickle["spatial_size"]
hist_bins = dist_pickle["hist_bins"]
hog_channel = "ALL" # Can be 0, 1, 2, or "ALL"

test_images = glob.glob('./test_images/test*.jpg')
f,ax = plt.subplots(3,2,figsize=(18,12))
ax = ax.ravel()
f.tight_layout()

# for i, img in enumerate(test_images):
#     vd = VehicleDetector()
#     output_img = vd.vehicle_detection_pipeline(mpimg.imread(img))
#     ax[i].imshow(output_img)
#     ax[i].set_axis_off()
#    
# plt.show()

test_video_output = 'video_output/test_video_out17_YUV20_avg10b2_threshOnAvg.mp4'
vd = VehicleDetector()
       
clip1 = VideoFileClip("test_video.mp4")
test_clip = clip1.fl_image(vd.vehicle_detection_pipeline) #NOTE: this function expects color images!!
test_clip.write_videofile(test_video_output, audio=False)