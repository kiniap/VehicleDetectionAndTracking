'''
Created on Dec 2, 2017

@author: kiniap
'''
'''
Created on Dec 2, 2017

@author: kiniap
'''
import cv2
import numpy as np
import matplotlib.pyplot as plt
import random
import glob
from mpl_toolkits.mplot3d import Axes3D

'''
Function to do a 3d plot of images in the different color spaces
'''
def plot3d(pixels, colors_rgb, ax,
        axis_labels=list("RGB"), axis_limits=((0, 255), (0, 255), (0, 255))):
    """Plot pixels in 3D."""

    # Create figure and 3D axes
    #ax = Axes3D(fig)

    # Set axis limits
    ax.set_xlim(*axis_limits[0])
    ax.set_ylim(*axis_limits[1])
    ax.set_zlim(*axis_limits[2])

    # Set axis labels and sizes
    ax.tick_params(axis='both', which='major', labelsize=14, pad=8)
    ax.set_xlabel(axis_labels[0], fontsize=16, labelpad=16)
    ax.set_ylabel(axis_labels[1], fontsize=16, labelpad=16)
    ax.set_zlabel(axis_labels[2], fontsize=16, labelpad=16)

    # Plot pixel values with colors given in colors_rgb
    ax.scatter(
        pixels[:, :, 0].ravel(),
        pixels[:, :, 1].ravel(),
        pixels[:, :, 2].ravel(),
        c=colors_rgb.reshape((-1, 3)), edgecolors='none')

    return ax  # return Axes3D object for further manipulation


'''
Convert a random car image to different color spaces
'''
car_images = glob.glob('vehicles/*/*.png')
non_car_images = glob.glob('non-vehicles/*/*.png')

# Read a car image
r = random.randint(0, len(car_images))
car_image = cv2.imread(car_images[r])
    
# Select a small fraction of pixels to plot by subsampling it
scale = max(car_image.shape[0], car_image.shape[1], 64) / 64  # at most 64 rows and columns
car_image_small = cv2.resize(car_image, (np.int(car_image.shape[1] / scale), np.int(car_image.shape[0] / scale)), interpolation=cv2.INTER_NEAREST)

# Convert subsampled image to desired color space(s)
car_image_small_RGB = cv2.cvtColor(car_image_small, cv2.COLOR_BGR2RGB)  # OpenCV uses BGR, matplotlib likes RGB
car_image_small_HSV = cv2.cvtColor(car_image_small, cv2.COLOR_BGR2HSV)
car_image_small_HLS = cv2.cvtColor(car_image_small, cv2.COLOR_BGR2HLS)
car_image_small_LUV = cv2.cvtColor(car_image_small, cv2.COLOR_BGR2LUV)
car_image_small_YUV = cv2.cvtColor(car_image_small, cv2.COLOR_BGR2YUV)
car_image_small_rgb = car_image_small_RGB / 255.  # scaled to [0, 1], only for plotting

'''
Convert a random non car image to different color spaces
'''
# Read a non car  image
r = random.randint(0, len(non_car_images))
non_car_image = cv2.imread(non_car_images[r])
    
# Select a small fraction of pixels to plot by subsampling it
scale = max(non_car_image.shape[0], non_car_image.shape[1], 64) / 64  # at most 64 rows and columns
non_car_image_small = cv2.resize(non_car_image, (np.int(non_car_image.shape[1] / scale), np.int(non_car_image.shape[0] / scale)), interpolation=cv2.INTER_NEAREST)

# Convert subsampled image to desired color space(s)
non_car_image_small_RGB = cv2.cvtColor(non_car_image_small, cv2.COLOR_BGR2RGB)  # OpenCV uses BGR, matplotlib likes RGB
non_car_image_small_HSV = cv2.cvtColor(non_car_image_small, cv2.COLOR_BGR2HSV)
non_car_image_small_HLS = cv2.cvtColor(non_car_image_small, cv2.COLOR_BGR2HLS)
non_car_image_small_LUV = cv2.cvtColor(non_car_image_small, cv2.COLOR_BGR2LUV)
non_car_image_small_YUV = cv2.cvtColor(non_car_image_small, cv2.COLOR_BGR2YUV)
non_car_image_small_rgb = non_car_image_small_RGB / 255.  # scaled to [0, 1], only for plotting

'''
Show the side by side images of the car and non car images in a certain color space
'''
# Plot and show the car and non car image side by side
fig = plt.figure(figsize=(16,8))
ax = fig.add_subplot(1,2,1, projection='3d')
ax1 = plot3d(car_image_small_YUV, non_car_image_small_rgb, ax, axis_labels=list("YUV"))
ax1.set_title('car image')

ax = fig.add_subplot(1, 2, 2, projection='3d')
ax2 = plot3d(non_car_image_small_YUV, non_car_image_small_rgb, ax, axis_labels=list("YUV"))
ax2.set_title('non car image')

plt.show()