#!/usr/bin/env python
# encoding: utf-8
"""
preprocessing.py

Created by Marcel Ruegenberg on 2013-06-04.

Preprocessing for the wafer dataset.
"""

import Image
import ImageDraw
import numpy as np
from util import as_ary

def align_rotations(images):
    """Align the rotations of a set of images by trying rotations up to 10 degrees clockwise and counterclockwise."""
    
    # Rotate the images to be all aligned.
    # We take the first image as the base, and then search in the -10 to +10 degrees for the rotation
    # 
    # We take the rotation angle for which the inner square of the images is best aligned
    
    image_shape = images[0].size
    
    aligned_images = []
    aligned_images.append(images[0])
    rotation_base_img = images[0]
    rotation_crop_x1 = int(image_shape[0] * (3/8.0))
    rotation_crop_x2 = image_shape[0] - rotation_crop_x1
    rotation_crop_y1 = 0 #  image_shape[1] / (3/8.0)
    rotation_crop_y2 = image_shape[1] - rotation_crop_y1
    rotation_crop_box = (rotation_crop_x1, rotation_crop_y1, rotation_crop_x2, rotation_crop_y2)
    rotation_test_angles = np.hstack([np.arange(0,3,0.25), np.arange(3.5,5,0.5),np.arange(5,10,1)])
    rotation_test_angles = np.hstack([0 - rotation_test_angles, rotation_test_angles])

    rotation_base = as_ary(rotation_base_img.crop(rotation_crop_box))
    for i, img in enumerate(images[1:]):
        min_diff = np.Inf
        min_diff_angle = np.nan
        best_rotated_image = None
        # find the angle a for which the difference between the original and the rotated image is minimized
        for a in rotation_test_angles:
            rotated_img = img.rotate(a, expand=0) # expand=0 to keep the size the same
            rotated_img_ary = as_ary(rotated_img.crop(rotation_crop_box)) # only compare cropped versions of the images. Restricting to a vertical bar
            diff = np.square(rotation_base - rotated_img_ary) 
            summed_diff = np.mean(diff)
            # print "Trying angle %f with diff %f" % (a, summed_diff)
            if summed_diff < min_diff:
                min_diff = summed_diff
                min_diff_angle = a
                best_rotated_image = rotated_img
        if best_rotated_image != None: # should always hold
            aligned_images.append(best_rotated_image)
            
    return aligned_images
    
def crop_to_wafer(image):
    """
    Crop the image to the actual wafer.
    This assumes that the wafer images are just cropped circles, with surrounding black.
    
    This also cuts away some of the margin.
    
    The result is, like the dataset itself, an RGB image with black background.
    The code could easily be modified to return an RGBA image with transparent background.
    
    The resulting image is slightly smaller than the original, but keeps aspect ratio.
    """
    
    size = image.size
    cutoff = 100 # 8
    
    box = (cutoff,cutoff,size[0]-cutoff,size[1]-cutoff)
    circle_image = Image.new("L", size, "black")
    draw = ImageDraw.Draw(circle_image)
    draw.ellipse(box,fill="white")
    del draw
    
    background_image = Image.new("RGB", size)
    
    background_image.paste(image,None,circle_image)
    
    return background_image.crop(box)
    