"""
Author: Marcel Ruegenberg

This method just takes a batch of images, calculates their mean and variance.
Then, pixels in all images are thresholded and and error detected if a pixel deviates from the mean by more then 2 times the variance.

According to Sebastian, this algorithm is what is used already by TI.
Note: Possibly, the algorithm actually uses the standard deviation instead of the variance.
      At least, that makes somewhat more sense and is basically a standard approach.
      
      I've also added some other things that make sense, like rotating the wafers to be aligned first.
"""

import Image
import ImageFilter
import os
import numpy as np

from preprocessing import align_rotations, crop_to_wafer
from util import as_ary
    
def to_img(dat,shape):
    return Image.fromarray(np.reshape(np.asarray(dat, dtype=np.uint8), shape))
    
def to_grayscale(img_as_array, shape):
    pixels = np.arange(shape[0] * shape[1]) * 3
    gray = np.amax(np.vstack([img_as_array[pixels], img_as_array[pixels + 1], img_as_array[pixels + 2]]),0)
    return gray
    
print "Loading images"
# Find all jpg images that are in the current directory. (Subdirectories are ignored.)
images = []
image_names = []
for dirname, dirnames, filenames in os.walk('.'):
    for filename in filenames:
        if filename.endswith(".jpg") or filename.endswith(".jpeg") or filename.endswith(".JPG"):
            img = Image.open(filename)
            
            sz = img.size
            img = img.resize((sz[0] / 2, sz[1] / 2), Image.NEAREST)
            
            img = img.filter(ImageFilter.MedianFilter()).filter(ImageFilter.SMOOTH)
            
            # # to grayscale
            # image_ary = as_ary(img)
            # pixels = np.arange(image_ary.shape[0] * image_ary.shape[1]) * 3
            # image = image_ary.flatten()
            # img = np.amax(np.vstack([image[pixels], image[pixels + 1], image[pixels + 2]]),0)
            # img = to_img(img, image_ary.shape[0:2])
            
            images.append(img)
            image_names.append(filename)
        
print "Aligning rotations"
aligned_images = align_rotations(images)

print "Cropping"
cropped_images = [crop_to_wafer(img) for img in aligned_images]

# Convert it all to one large array:
image_arr = np.vstack([as_ary(img).flatten() for img in cropped_images])

image_shape = as_ary(cropped_images[0]).shape

print "Computing mean and variance"
# Compute mean and variance of the batch
mean = np.mean(image_arr, axis=0)
std  = np.std(image_arr,axis=0)
threshold = 2


# rescale to [0.0, 255]
def rescale(dat):
    mn = np.amin(dat)
    mx = np.amax(dat)
    return ((dat - mn) / (mx - mn)) * 512

# mean_img = to_img(mean, image_shape)
# mean_img.show()

# to_img(var0 * 5, image_shape).show()
# to_img(std * 10, image_shape).show()
# to_img(to_grayscale(std, image_shape) * 10, image_shape[0:2]).show()


print "Finding bad images"
bad_imgs = []
for i, img in enumerate(image_arr):
    sqdiff = np.sqrt(np.square(mean - img))
    print "Max: ", np.amax(sqdiff)
    
    # sqdiff_merged = to_grayscale(sqdiff, image_shape)
    have_bad = np.any(sqdiff > threshold * std)

    if have_bad:
        # to_img(sqdiff_merged * 5, image_shape[0:2]).show()
        
        # to_img(img, image_shape).show()
        bad_pxs = np.where(sqdiff > threshold * std)[0]
        
        img_data = np.asarray(image_arr[i], dtype=np.uint8)
        
        # img_data[bad_pxs * 3 + 0] = 255
        # img_data[bad_pxs * 3 + 1] = 0
        # img_data[bad_pxs * 3 + 2] = 0

        # # # Dye the offending pixels red.
        # # # (x / 3) * 3 rounds to multiples of 3
        img_data[(bad_pxs / 3) * 3] = 255
        img_data[(bad_pxs / 3) * 3 + 1] = 0
        img_data[(bad_pxs / 3) * 3 + 2] = 0
        
        img = Image.fromarray(np.reshape(img_data, image_shape))
        img.show()
        
        bad_imgs.append(img)
        
