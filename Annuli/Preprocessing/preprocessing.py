#!/usr/bin/env python
# encoding: utf-8
"""
preprocessing.py

Created by Marcel Ruegenberg on 2013-06-05.
"""

import Image
import numpy as np

import ImageFilter
import ImageChops
import ImageOps
import scipy.ndimage as ndi
import scipy.misc

import sys
import os

def find_neighbors(pos, size):
    i,j = pos
    neighbors = []
    for i0 in [i-1,i,i+1]:
        if i0 < 0 or i0 >= size[0]:
            continue
        for j0 in [j-1,j,j+1]:
            if j0 < 0 or j0 >= size[1]:
                continue
            if i0 == i and j0 == j:
                continue
                
            neighbors.append((i0,j0))
            
    return neighbors

def extract_ultrasound_pixels(image):
    """
    Returns a bitmask of the same size as the image with the pixels where there is ultrasound data white, all others black.
    The white region is guaranteed to be one connected block of pixels.
    
    :param image The image on which to segment the ultrasound data
    :return an image of mode "1" in the PIL library, aka a bit mask
    """
    
    # IMPORTANT: for various images, different variants of these two values are necessary.
    #            If you get an image, and cleaning doesn't work, change these values to the commented-out variant.
    # Options / tuning of this function:
    hole_close_radius = 11 # 37
    expand_mask = True # False
    
    # The ultrasound images have the following characteristics:
    # Due to the way ultrasound works, the image is usually a cropped version of an arc over 90 degrees whose center is at the top:
    #             **
    #            ****
    #          ********
    #        ************
    #           ******
    #     
    # In our dataset, there is sometimes a green line at the bottom as well.
    #
    # Additionally, the data contain jpeg artifacts.

    img_orig_data = image.load()
    size = image.size
    
    # Untint, so as to make most pixels grayscale
    # First, find the tint color. We do that by taking the average of a 20x20 pixels sample on the center of the image:
    istart = size[0] / 2 - 10
    jstart = size[1] / 2 - 10
    n = 0
    avg = np.zeros(3)
    for i in xrange(istart,istart+20):
        for j in xrange(jstart,jstart+20):
            px = img_orig_data[i,j]
            px_channels = np.asarray(list(px))
            avg = avg + px_channels
            n = n+1
    avg = avg / n
    avg_r = int(avg[0])
    avg_g = int(avg[1])
    avg_b = int(avg[2])
    gr = (avg_r + avg_g + avg_b) / 3
    # Remove tint:
    for i in xrange(size[0]):
        for j in xrange(size[1]):
            px = img_orig_data[i,j]
            mx = max(max(px[0],px[1]),px[2])
            mn = min(min(px[0],px[1]),px[2])
        
            if mx - mn < 5:
                av = (px[0] + px[1] + px[2]) / 3
                img_orig_data[i,j] = (av,av,av)
            else:
                img_orig_data[i,j] = ((px[0] - avg_r) + gr, (px[1] - avg_g) + gr, (px[2] - avg_b) + gr)
                
    # do some filtering / blurring. Median and minimum filters are good for getting rid of JPEG artifacts.
    img_data_min = image.filter(ImageFilter.MinFilter(size=1)).load()
    img_data_max = image.filter(ImageFilter.MaxFilter(size=1)).load()
    
    # The mask to return. Contains only 1 bit per pixel
    mask = Image.new("1",size,"white") # We don't use a Python array because we want PIL's image filtering functions
    mask_access = mask.load()
    
    # If the bottom two rows are mostly black, we want those in our mask, in order to get those ultrasound curves that are basically gray
    non_black_cnt = 0
    for i in xrange(size[0]):
        for j in xrange(size[1] - 2, size[1]):
            px = img_data_min[i,j]
            if px[0] >= 30 or px[1] >= 30 or px[2] >= 30:
                non_black_cnt = non_black_cnt + 1
                
    if not non_black_cnt < 30:
        non_black_cnt = 0
        for i in xrange(size[0]):
            for j in xrange(size[1] - 6, size[1] - 4):
                px = img_data_min[i,j]
                if px[0] >= 30 or px[1] >= 30 or px[2] >= 30:
                    non_black_cnt = non_black_cnt + 1
                    
    if not non_black_cnt < 30:
        non_black_cnt = 0
        for i in xrange(size[0]):
            for j in xrange(size[1] - 30, size[1] - 28):
                px = img_data_min[i,j]
                if px[0] >= 30 or px[1] >= 30 or px[2] >= 30:
                    non_black_cnt = non_black_cnt + 1
    
    if non_black_cnt < 30:
        for i in xrange(size[0]):
            for j in xrange(size[1] - 2, size[1]):
                mask_access[i,j] = 0
                

    # Find high-contrast places, as those are not typical of ultrasound either and therefore hint at artifacts
    for i in xrange(size[0]):
        for j in xrange(size[1]):
            px = img_orig_data[i,j] # do not use img_data here, since that might not contain smaller white patches any longer
            px_channels = np.asarray(list(px))
    
            mn = min(min(px[0],px[1]),px[2])
            if mn >= 250:
                neighbors = find_neighbors((i,j),size)
    
                add_px = False
                # find out if some neighbor pixel is almost black
                for i0,j0 in neighbors:
                    px_n = img_data_min[i0,j0] # yes, here we use img_data.
                    mx_n = max(max(px_n[0],px_n[1]),px_n[2])
                    if mx_n < 12:
                        mask_access[i0,j0] = 0 # also add the neighbors that lead to the px being added
                        add_px = True
                        
                if add_px:
                    mask_access[i,j] = 0
                    for i0,j0 in neighbors:
                        px_n = img_data_max[i0,j0]
                        mn_n = min(min(px_n[0],px_n[1]),px_n[2])
                        if mn_n >= 240:
                            mask_access[i0,j0] = 0


    # Find pixels that are not basically gray by looking for those with high relative variance between their three channels
    # do some filtering / blurring. Median filters are good for getting rid of JPEG artifacts.
    image = image.filter(ImageFilter.MedianFilter()).filter(ImageFilter.MedianFilter())
    img_data = image.load()
    # img_data = image.filter(ImageFilter.GaussianBlur(radius=3)).load()
    for i in xrange(size[0]):
        for j in xrange(size[1]/2,size[1]):
            px = img_data[i,j]
            px_channels = np.asarray(list(px))
    
            # find non-gray pixels
            mx = np.max(px_channels)
            if mx > 7:
                mx = min(mx,50)
                relative_variance = np.var(px_channels) / mx
                
                if relative_variance > 5:
                    mask_access[i,j] = 0
                    
    # Second, do some processing to remove artifacts from the result
    mask = mask.filter(ImageFilter.MedianFilter()) # denoise
    mask = mask.filter(ImageFilter.MinFilter(size=5)) # expand
    
    mask = mask.filter(ImageFilter.MinFilter(size=3)) # expand & contract to close holes
    mask = mask.filter(ImageFilter.MaxFilter(size=3))
    
    mask_access = mask.load()
    
    work_img = ImageOps.autocontrast(ImageChops.multiply(image,ImageChops.invert(Image.fromarray(ndi.gaussian_filter(np.asarray(image),3)))))
    work_img_data = work_img.load()
    # work_img.show()
    
    # Expand the area by almost-black and colored pixels using breadth-first search
    px_queue = []
    for i in xrange(size[0]):
        for j in xrange(size[1]):
            if mask_access[i,j] == 0:
                px_queue.append((i,j))
                
    while px_queue:
        pos = px_queue.pop(0)
        
        neighbors = find_neighbors(pos,size)
                  
        for i,j in neighbors:
            if mask_access[i,j] == 0:
                continue
            px = work_img_data[i,j]
            mx = max(max(px[0],px[1]),px[2])
            if mx < 5:
                mask_access[i,j] = 0
                px_queue.append((i,j))
            else:
                px_channels1 = np.asarray(list(px))
                
                px2 = img_data[i,j]
                px_channels2 = np.asarray(list(px2))
                
                mx2 = max(max(px2[0],px2[1]),px2[2])                
                
                if mx > 7 or mx2 > 7:
                    mx = min(mx,30)
                    mx2 = min(mx2,30)
                    relative_variance = np.var(np.asarray(list(px_channels))) / mx
                    relative_variance2 = np.var(np.asarray(list(px_channels2))) / mx2
                    
                    if relative_variance > 5 or relative_variance2 > 2:
                        mask_access[i,j] = 0
                        px_queue.append((i,j))
                

    # Expand & contract to close holes. This step is needed because some ultrasound lines are actually gray
    # FIXME: these might in some cases lead to undesirable results (i.e removing too much)
    mask = mask.filter(ImageFilter.MinFilter(size=hole_close_radius)) 
    mask = mask.filter(ImageFilter.MaxFilter(size=hole_close_radius))
    del mask_access
    
    # IMPORTANT: this part is needed for some images, and not for others.
    if expand_mask:    
        # Expand the area in the mask to the corners, i.e we assume that the actual ultrasound data is in a connected region somewhere in the "middle" of the image
        inner_region = mask.copy()
        inner_region_access = inner_region.load()
    
        init_pos = (size[0]/2,size[1]/2)
        px_queue = [init_pos]
        inner_region_access[init_pos] = 0
        while px_queue:
            pos = px_queue.pop(0)
            neighbors = find_neighbors(pos,size)
            for pos_n in neighbors:
                if inner_region_access[pos_n] != 0:
                    inner_region_access[pos_n] = 0
                    px_queue.append(pos_n)
                
        inner_region = ImageChops.invert(inner_region)
    
        mask = ImageChops.multiply(mask, inner_region)
        
    mask_access = mask.load()
    
    for j in range(size[1] - 50,size[1]):
        masked_in_line = 0
        for i in range(size[0]):
            if mask_access[(i,j)] == 0:
                masked_in_line = masked_in_line + 1
                
        if masked_in_line > 0.6 * size[1]:
            for i in range(size[0]):
                px = img_data[(i,j)]
                mx = max(max(px[0],px[1]),px[2])
                if mask_access[(i,j)] == 0 or mx < 5:
                    masked_in_line = masked_in_line + 1
        if masked_in_line > 0.8 * size[1]:
            for i in range(size[0]):
                mask_access[(i,j)] = 0

    return mask

def preprocessing(datapath,debug=False,no_replace=False):
    img_names = []
    for dirname, dirnames, filenames in os.walk(datapath):
        for filename in filenames:
            if filename.endswith(".jpg") or filename.endswith(".jpeg") or filename.endswith(".JPG"):
                img_names.append(filename)

    for img_name in img_names:
        try:
            print "Cleaning %s" % img_name

            img = Image.open(img_name)
            if img.getbands() != ("R","G","B"):
                print "Skipping non-RGB image (%s instead of RGB)" % img.getbands()
                continue
            mask = extract_ultrasound_pixels(img)
            if debug:
                blackened_img = Image.new("RGB", img.size, "red")
                blackened_img.paste(img,(0,0),mask)
            else:
                blackened_img = Image.new("L", img.size, "black")
                blackened_img.paste(img,(0,0),mask)

            if debug or no_replace:
                filenm, ext = os.path.splitext(img_name)
                savenm = filenm + ".cleaned" + ext
            else:
                savenm = img_name

            blackened_img.save(savenm)
        except IOError:
            pass
        
if __name__ == '__main__':
    no_replace = "--no-replace" in sys.argv
    debug = "--debug" in sys.argv
    
    preprocessing('.',debug,no_replace)
