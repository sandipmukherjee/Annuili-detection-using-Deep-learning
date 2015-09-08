#!/usr/bin/env python
# encoding: utf-8
"""
generate_datasets.py

Created by Marcel Ruegenberg on 2013-06-23.
"""

import Image
import numpy as np
import cPickle as pickle

import sys
import os


def get_img_size(file_name):
    img = Image.open(file_name)
    return img.size
    
def main():
    img_names = []
    # TODO: use `argparse` (see http://docs.python.org/2/library/argparse.html)
    two_ch = "--2ch" in sys.argv
    left = "--left" in sys.argv
    right = "--right" in sys.argv
    four_ch = "--4ch" in sys.argv
    align_left_right = "--align" in sys.argv
    
    unbalanced_test = "--unbalanced-test" in sys.argv
    
    # use multiple classes?
    # if yes, neither or both of --left and --right should be specified
    multi_classes = "--multiclass" in sys.argv
    
    numpy_dump = "--use-numpy" in sys.argv # use NumPy's own dumping facilities to build a .npz file?
    
    out_file = "dataset.pkl"
    if "--out" in sys.argv:
        i = sys.argv.index("--out") + 1
        if i < len(sys.argv) - 1: # out_file must come before data dir, hence len - 1
            out_file = sys.argv[i]
        else:
            print "Please specify a filename after `--out`."
            
    patch_size = 20
    if "--patch-size" in sys.argv:
        i = sys.argv.index("--patch-size") + 1
        if i < len(sys.argv) - 1:
            patch_size = int(sys.argv[i])
            
    print "Patch size: ", patch_size
    
    if not two_ch and not four_ch: # if no options are set, take whole dataset
        two_ch = True
        four_ch = True
    if left and right: # if both should be taken, there is no restriction from the left/right option
        left = False
        right = False
    if two_ch:
        print("Generating 2 chamber data")       
    if four_ch:
        print("Generating 4 chamber data")
        
    if left or (not left and not right):
        print("Generating left")
    if right or (not left and not right):
        print("Generating right")
        
    if multi_classes:
        print "Generating 3-class data (no annulus, left, right)"
        
    if align_left_right:
        if (left and right) or (not left and not right):
            print "Aligning left / right patches by mirroring"
        elif multi_classes: # aligning is not a good idea when generating multiple classes
            print "Warning: Generating multi-class data, but got instructed to align left and right patches. Ignoring."
            align_left_right = False
        else:
            print "Warning: Got instructed to align left and right patches, but only working on left or right. Ignoring."
            align_left_right = False
        
    data_dir = sys.argv[-1]
    if not data_dir[-1] == os.path.sep:
        data_dir = data_dir + os.path.sep
        
    patch_offset = patch_size / 2 #  / 4 # the amount that the overlapping patches are offset from one to the next
        
    for dirname, dirnames, filenames in os.walk(data_dir):
        for filename in filenames:
            if filename.endswith(".jpg") or filename.endswith(".jpeg") or filename.endswith(".JPG"):
                if (((two_ch and "2ch" in filename) or (four_ch and "4ch" in filename)) and
                    ((not left or "_l." in filename) and (not right or "_r." in filename))):
                    img_names.append(data_dir + filename)
                    
    del filenames, filename
                    
    size = get_img_size(img_names[0])
    flat_size = size[0] * size[1]
    
    # Split into train/test/validation set along 60/20/20:
    print("Shuffling image names")

    np.random.seed(42) # use repeatable seeds
    img_names = np.random.permutation(img_names)
    
    idx1 = int(len(img_names) * 0.6)
    idx2 = int(len(img_names) * 0.8)
    idx3 = int(len(img_names))
    img_names_train = img_names[0:idx1]
    img_names_valid = img_names[idx1:idx2]
    img_names_test = img_names[idx2:idx3]
    
    del img_names
     
    print("Generating negative positions")
    center_box_origin = ((size[0] - patch_size) / 2, (size[1] - patch_size) / 2)
    i_coords = [x * patch_offset for x in range(int((size[0] - patch_offset) / float(patch_offset)))]
    j_coords = [x * patch_offset for x in range(int((size[1] - patch_offset) / float(patch_offset)))]
    def overlaps_center_box(i,j):
        return (i + patch_size > center_box_origin[0] and j + patch_size > center_box_origin[1] and
                i <= center_box_origin[0] + patch_size / 2 and j <= center_box_origin[1] + patch_size / 2)
    neg_positions = [(i,j) for i in i_coords for j in j_coords if not overlaps_center_box(i,j)]
    neg_count = len(neg_positions)
    print "Neg: ", neg_count
    
    
    pos_count = 3 * 3
    pos_count_test = 1
    
    # to disable balancing, just set both of these to 0
    annulus_dup_count = int((neg_count / (pos_count * 1.0)) * pos_count)
    annulus_dup_mul = int(neg_count / pos_count)
    
    if multi_classes:
        annulus_dup_count = annulus_dup_count + annulus_dup_mul * pos_count
        annulus_dup_mul = annulus_dup_mul * 2
    
    print "Pos: ", (pos_count, annulus_dup_count, annulus_dup_mul)
    print "Pos test: ", pos_count_test
    
    
    train_count = len(img_names_train) * (annulus_dup_count + neg_count)
    valid_count = len(img_names_valid) * (annulus_dup_count + neg_count)
    if unbalanced_test:
        test_count  = len(img_names_test)  * (pos_count_test    + neg_count)
    else:
        test_count  = len(img_names_test)  * (annulus_dup_count + neg_count)
    
    training_set   = np.zeros((train_count, patch_size * patch_size + 1), dtype=np.uint8)
    validation_set = np.zeros((valid_count, patch_size * patch_size + 1), dtype=np.uint8)
    test_set       = np.zeros((test_count,  patch_size * patch_size + 1), dtype=np.uint8)
    
    global indx, worked_imgs
    indx = 0
    worked_imgs = 0
    
    
    def handle_img(img_name,target_list,test=False):
        global worked_imgs, indx
        if worked_imgs % 100 == 0:
            print("Working on %s" % img_name)
        elif worked_imgs % 25 == 0:
            print "."
        worked_imgs = worked_imgs + 1
        img = Image.open(img_name)
        if img.mode != "L":
            print("WARNING: Got image with mode %s instead of L! Are you using the cleaned dataset?" % img.mode)
            
        if align_left_right and "_r." in img_name:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        
        if test:
            local_pos_count = pos_count_test
        else:
            local_pos_count = pos_count
        
        annulus_samples = np.zeros((local_pos_count,patch_size * patch_size + 1), dtype=np.uint8)
            
        indx_local = 0
        
        pos_positions = []
        if not test:
            for k in range(3):
                for l in range(3):
                    k_pos = np.random.randint(-5,5)
                    l_pos = np.random.randint(-5,5)
                    pos_positions.append((center_box_origin[0] + k_pos, center_box_origin[1] + l_pos))
        else:
            pos_positions.append((center_box_origin[0] + np.random.randint(-5,5), center_box_origin[1] + np.random.randint(-5,5)))
            
        if test:
            dup_mul   = 1
            dup_count = local_pos_count
        else:
            dup_mul   = annulus_dup_mul
            dup_count = annulus_dup_count
        
        class_number = 1.
        if multi_classes and "_r." in img_name:
            class_number = 2.
        
        for (i,j) in pos_positions:
            annulus = img.crop((i,j,i+patch_size, j + patch_size))
            sample = np.hstack([np.asarray(annulus,dtype=np.uint8).flatten(), class_number])
            # annulus.show()
            
            annulus_samples[indx_local] = sample
            indx_local = indx_local + 1
            
            for k in range(dup_mul): # just duplicate all annulus samples
                target_list[indx] = sample
                indx = indx + 1
        
        for k in range((local_pos_count * dup_mul), dup_count): # for additional balancing, add some samples again
            target_list[indx] = annulus_samples[np.random.randint(annulus_samples.shape[0])]
            indx = indx + 1
        
        for (i,j) in neg_positions:
            non_annulus = img.crop((i,j,i+patch_size, j + patch_size))
            # non_annulus.show()
            target_list[indx] = np.hstack([np.asarray(non_annulus,dtype=np.uint8).flatten(), np.zeros(1)])
            indx = indx + 1
            
    print "Generating training set"
    for img_name in img_names_train:
        handle_img(img_name,training_set,test=False)
        
    if indx != train_count:
        print "Expected %i training set samples, but got %i!" % (train_count, indx)
        
    print "Generating validation set"
    indx = 0
    for img_name in img_names_valid:
        handle_img(img_name,validation_set,test=False)
        
    if indx != valid_count:
        print "Expected %i validation set samples, but got %i!" % (train_count, indx)
     
    print "Generating test set"
    indx = 0
    for img_name in img_names_test:
        handle_img(img_name,test_set,test=unbalanced_test)

    if indx != test_count:
        print "Expected %i validation set samples, but got %i!" % (train_count, indx)
        
    class_labels = ["no annulus"]
    if multi_classes:
        class_labels = class_labels + ["left annulus", "right annulus"]
    else:
        class_labels.append("annulus")
    
    print("Dumping")
    
    if numpy_dump:
        out_file_name, out_file_ext = os.path.splitext(out_file)
        np.savez_compressed(out_file_name, training_set, validation_set, test_set, class_labels)
    else:
        with open(out_file,"wb") as f:
            pickle.dump((training_set,validation_set,test_set,class_labels), f, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    main()

