#!/usr/bin/env python
# encoding: utf-8
"""
preprocessing.py

Created by Marcel Ruegenberg on 2013-06-04.

Preprocessing for the wafer dataset.
"""

import Image
import ImageDraw
import ImageChops
import ImageFilter
import ImageEnhance
import numpy as np

import sys
import os
import cPickle

import csv

from subprocess import call



def as_ary(img):
    return np.asarray(img,np.uint8)
    
def subtract_blur(img):
    diff = ImageChops.difference(img, img.filter(ImageFilter.GaussianBlur(radius=20)))
    return diff

def align_rotations(images):
    """
    Align the rotations of a set of images by trying rotations up to 10 degrees clockwise and counterclockwise.
    :param images A list or array of images
    :return The list of tuples of aligned images with corresponding rotations
    """
    
    if not images:
        return []
        
    # Rotate the images to be all aligned.
    # We take the first image as the base, and then search in the -10 to +10 degrees for the rotation
    # 
    # We take the rotation angle for which the inner square of the images is best aligned
    
    image_shape = images[0].size
    
    aligned_images = []
    aligned_images.append((images[0],0))
    rotation_base_img = images[0]
    rotation_base_img = subtract_blur(rotation_base_img)

    rotation_crop_x1 = int(image_shape[0] * (3/8.0))
    rotation_crop_x2 = image_shape[0] - rotation_crop_x1
    rotation_crop_y1 = int(image_shape[1] * (1/8.0))
    rotation_crop_y2 = image_shape[1] - rotation_crop_y1
    rotation_crop_box = (rotation_crop_x1, rotation_crop_y1, rotation_crop_x2, rotation_crop_y2)
    rotation_test_angles = np.hstack([np.arange(0,3,0.25), np.arange(3.5,5,0.5),np.arange(5,10,1)])
    rotation_test_angles = np.hstack([0 - rotation_test_angles, rotation_test_angles])

    rotation_base = as_ary(rotation_base_img.crop(rotation_crop_box))
    for i, img in enumerate(images[1:]):
        clear_img = subtract_blur(img)
        base_diff = np.mean(np.square(rotation_base - as_ary(clear_img.crop(rotation_crop_box))))
        
        min_diff = np.Inf
        min_diff_angle = np.nan
        best_rotated_image = None
        # find the angle a for which the difference between the original and the rotated image is minimized
        for a in rotation_test_angles:
            rotated_img = clear_img.rotate(a, expand=0) # expand=0 to keep the size the same
            rotated_img_ary = as_ary(rotated_img.crop(rotation_crop_box)) # only compare cropped versions of the images. Restricting to a vertical bar
            diff = np.square(rotation_base - rotated_img_ary) 
            summed_diff = np.mean(diff)
            # print "Trying angle %f with diff %f" % (a, summed_diff)
            if summed_diff < min_diff:
                min_diff = summed_diff
                min_diff_angle = a
                best_rotated_image = rotated_img
        
        if min_diff < 0.97 * base_diff: # by default, don't rotate => the improvement from rotating has to be somewhat significant
            aligned_images.append((img.rotate(min_diff_angle, expand=0),min_diff_angle))
        else:
            aligned_images.append((img,0))
            
    return aligned_images
    
    
def crop_to_wafer(image,cutoff=10):
    """
    Crop the image to the actual wafer.
    This assumes that the wafer images are just cropped circles, with surrounding black.
    
    This also cuts away some of the margin.
    
    The result is, like the dataset itself, an RGB image with black background.
    The code could easily be modified to return an RGBA image with transparent background.
    
    The resulting image is slightly smaller than the original, but keeps aspect ratio.
    
    :param image The image to crop
    :param cutoff How many pixels to cut off at each side
    :return The cropped image
    """
    
    size = image.size
    
    box = (cutoff,cutoff,size[0]-cutoff,size[1]-cutoff)
    circle_image = Image.new("L", size, "black")
    draw = ImageDraw.Draw(circle_image)
    draw.ellipse(box,fill="white")
    del draw
    
    background_image = Image.new("RGB", size)
    
    background_image.paste(image,None,circle_image)
    
    return background_image.crop(box)
    
    
def preprocess_batch(chip_name,batch_name=None,data_dir='TI',masks_dir=None):
    """
    :param chip_name The name of the chip / the name of the directory which contains the batch
    :param batch_name The name of the actual batch / the directory that contains the "good" and "bad" subdirectories of images.
        If None, we just take the `chip_name` directory
    :param data_dir The directory where the directories for each batch are located
    :param masks_dir The directory where the mask images are located. If None, we assume `data_dir/../masks TI`.
    :return A 3-tuple with
        - a NumPy array of images
        - a NumPy array of masks with the same dimensions
        - a list of labels, where 0="not bad", 1="bad"
    """

    if not masks_dir:
        masks_dir = os.path.join(data_dir,os.pardir,"masks TI")

    data = os.path.join(data_dir,chip_name)
    masks = os.path.join(masks_dir,chip_name)
    have_masks = os.path.exists(masks)
    
    if not have_masks:
        print "Warning: no masks for batch found. Assuming black masks."
        

    if batch_name:
        data = os.path.join(data,batch_name)
        masks = os.path.join(masks,batch_name)

    result = []

    # Load images from a path, and ass a label to each of them
    def get_imgs(cur_path,cur_mask_path,is_bad):
        imgs = []
        masks = []
        labels = []
        names = []

        for dirname, dirnames, filenames in os.walk(cur_path):
            for filename in filenames:
                img = Image.open(os.path.join(cur_path,filename))
                if not have_masks:
                    mask = Image.new("RGB",img.size,"black")
                else:
                    mask_filename = (filename + ".mask.png")
                    mask = Image.open(os.path.join(cur_mask_path,mask_filename)).convert("L")

                imgs.append(img)
                masks.append(mask)
                labels.append(is_bad)
                names.append(os.path.splitext(filename)[0])

        return (imgs,masks,labels,names)

    (good_imgs,good_img_masks,good_img_labels,good_img_names) = get_imgs(os.path.join(data,'good'),os.path.join(masks,'good'),0)
    (bad_imgs,bad_img_masks,bad_img_labels,bad_img_names) = get_imgs(os.path.join(data,'bad'),os.path.join(masks,'bad'),1)

    # Since images are now labeled with is_bad, we can join everything
    all_imgs = good_imgs + bad_imgs
    all_masks = good_img_masks + bad_img_masks
    all_labels = good_img_labels + bad_img_labels
    all_names = good_img_names + bad_img_names
    
    # work around problem that dataset is not totally consisent:
    min_size = all_imgs[0].size 
    max_size = all_imgs[0].size
    for img in all_imgs:
        if img.size[0] < min_size[0]:
            min_size = img.size
        elif img.size[0] > max_size[0]:
            max_size = img.size
    if min_size != max_size:
        crop_box_upper_left = ((max_size[0] - min_size[0]) / 2, (max_size[1] - min_size[1]) / 2)
        
        for i in range(len(all_imgs)):
            if all_imgs[i].size != min_size:
                all_imgs[i] = all_imgs[i].resize(min_size,Image.ANTIALIAS)
                all_masks[i] = all_masks[i].resize(min_size,Image.ANTIALIAS)
            


    del good_imgs, bad_imgs
    del good_img_masks, bad_img_masks
    del good_img_labels, bad_img_labels
    del good_img_names, bad_img_names

    # Rotate the images to minimize variance
    aligned_images_and_angles = align_rotations(all_imgs)
    del all_imgs

    aligned_images = [t[0] for t in aligned_images_and_angles]
    align_angels = [t[1] for t in aligned_images_and_angles]
    del aligned_images_and_angles
    aligned_masks = []
    for i in range(len(align_angels)):
        a = align_angels[i]
        m = all_masks[i].rotate(a,expand=0)
        aligned_masks.append(m)

    del all_masks

    # Remove light corner pixels from wafer images
    cropped_images = [crop_to_wafer(img,cutoff=10) for img in aligned_images]
    del aligned_images
    cropped_masks  = [crop_to_wafer(img,cutoff=10) for img in aligned_masks]
    del aligned_masks

    return (cropped_images,cropped_masks,all_labels,all_names)


def preprocess_all_batches(save_dir,data_dir,masks_dir=None):
    """
    Load and preprocess all batches in `dir`, save them as picked data in `save_dir`.
    :param save_dir The directory to which to save the pickled preprocessed batches
    :data_dir The directory that contains the whole dataset
    :masks_dir Optional. For details see `load_batch`.
    """

    for fname in os.listdir(data_dir):
        if not os.path.isdir(os.path.join(data_dir,fname)):
            continue

        chip_name = fname
        print "Chip name: \t", chip_name
        
        def handle_batch(batch_name):
            if not os.path.exists(os.path.join(save_dir,chip_name)):
                os.mkdir(os.path.join(save_dir,chip_name))
            batch_dir = os.path.join(save_dir,chip_name,batch_name)
            if os.path.exists(batch_dir):
                if not batch_name == "":
                    return
            else:
                os.mkdir(batch_dir)

            print "\tBatch name:\t", batch_name
            (cropped_images,cropped_masks,all_labels,all_names) = preprocess_batch(chip_name, batch_name, data_dir, masks_dir)

            for i, filename in enumerate(all_names):
                cropped_images[i].save( os.path.join(batch_dir,filename + ".jpg"), "JPEG", quality=90 )
                cropped_masks[i].save( os.path.join(batch_dir,filename + "_mask.png"), "PNG" )
                
            call(["convert",os.path.join(batch_dir,"*.jpg"),"-evaluate-sequence","Median",os.path.join(batch_dir,"median.jpg")])

            with open(os.path.join(batch_dir,"labels.pkl"), "wb") as f:
                f.write(cPickle.dumps((all_labels,all_names),protocol=cPickle.HIGHEST_PROTOCOL))
        
        if chip_name.startswith("_"):
            handle_batch("")
        else:
            for fname1 in os.listdir(os.path.join(data_dir,chip_name)):
                if not os.path.isdir(os.path.join(data_dir,chip_name,fname1)):
                    continue
                batch_name = fname1
                handle_batch(batch_name)


def labels_to_csv(save_dir):
    for fname in os.listdir(save_dir):
        if not os.path.isdir(os.path.join(save_dir,fname)):
            continue

        chip_name = fname
        print "Chip name: \t", chip_name
        
        def handle_batch(batch_name):
            batch_dir = os.path.join(save_dir,chip_name,batch_name)
            csvfile = os.path.join(batch_dir,"labels.csv")
            if os.path.exists(csvfile):
                return
            with open(os.path.join(batch_dir,"labels.pkl"), "r") as f:
                (all_labels,all_names) = cPickle.load(f)
            with open(csvfile, "wb") as f:
                writer = csv.writer(f)
                for i in range(len(all_labels)):
                    writer.writerow([all_labels[i], all_names[i]])
        
        if chip_name.startswith("_"):
            handle_batch("")
        else:
            for fname1 in os.listdir(os.path.join(save_dir,chip_name)):
                if not os.path.isdir(os.path.join(save_dir,chip_name,fname1)):
                    continue
                batch_name = fname1
                handle_batch(batch_name)



def main():
    # labels_to_csv(sys.argv[2])
    preprocess_all_batches(sys.argv[2],sys.argv[1])

if __name__ == '__main__':
    main()
