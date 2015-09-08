__author__ = 'sandip'

from PIL import Image
import numpy as np
import ImageDraw
import cPickle
import ImageFilter
from random import shuffle

import sys
import os


def make_annuili_data():
    type1 = "000"
    type2 = "00"
    type3 = "0"
    chamber2 = "2ch"
    chamber4 = "4ch"
    left = "l"
    right = "r"
    suffix = ".jpg"
    imagePath = " "
    underscore = "_"
    images = []
    image_patches_2ch = []
    image_patches_2ch_label = []
    image_patches_4ch = []
    image_patches_4ch_label = []
    trainingSet = []
    trainingSet_shuffle = []
    trainingLabel = []
    trainingLabel_shuffle = []
    validationSet = []
    validationSet_shuffle = []
    validationLabel = []
    validationLabel_shuffle = []
    testingSet = []
    testingSet_shuffle = []
    testingLabel = []
    testingLabel_shuffle = []
    total_count = 0
    for dirname, dirnames, filenames in os.walk('.'):
        for filename in filenames:
            if "2ch" in filename:
                if filename.endswith(".jpg") or filename.endswith(".jpeg") or filename.endswith(".JPG"):
                    imagename = filename
                try:
                    print imagename
                    img = Image.open(imagename)
                    trainingSet,trainingLabel,validationSet,validationLabel,testingSet,testingLabel,image_patches_4ch,image_patches_4ch_label,total_count  = Annuili_and_non_annuili_patches(img,imagename,trainingSet,trainingLabel,validationSet,validationLabel,testingSet,testingLabel,image_patches_4ch,image_patches_4ch_label,total_count)

                    del img
                except IOError:
                    pass


    print "shuffling data.........."


    print len(trainingSet)
    print len(trainingLabel)

    print len(validationSet)
    print len(validationLabel)

    print len(testingSet)
    print len(testingLabel)
    print "Pickling 3 parts"
    #Generating training(60%), validation(20%) and testing(20%) split for 2 chamber data.

    data1 = [trainingSet,trainingLabel,validationSet,validationLabel,testingSet,testingLabel]
    cPickle.dump( data1, open( "AnnuiliData_2ch.pkl", "ab" ) )
    print "Data for 2 chamber images pickled to file AnnuiliData_2ch.pkl at the present directory"

def list_shuffle(input_list,output_list):
    list1_shuf = []
    list2_shuf = []
    index_shuf = range(len(input_list))
    shuffle(index_shuf)
    for i in index_shuf:
        list1_shuf.append(input_list[i])
        list2_shuf.append(output_list[i])

    return list1_shuf,list2_shuf




def Annuili_and_non_annuili_patches(image,filename,trainingSet,trainingLabel,validationSet,validationLabel,testingSet,testingLabel,image_patches_4ch,image_patches_4ch_label,total_count):
    patches_as_arrays = []
	
    training_count = 100000
    validation_count = 130000
    gray_image = image.convert("L")
    #Bulding patches from centres of the image to build positive annuili
    for left in range(85,100,5):
        for top in range(85,100,5):

            annuili_cropped_image = crop_image(gray_image,left,top).load()
            image_array = []
            for w in range(20):
                for h in range(20):
                    pixel = annuili_cropped_image[w,h]
                    image_array.append(pixel)
                    #annuili_image_matrix[w,h] = pixel
            if "2ch" in filename:
                if total_count < training_count:
                    trainingSet.append(image_array)
                    if "_l" in filename:
                        trainingLabel.append(1)
                    elif "_r" in filename:
                        trainingLabel.append(2)
                elif total_count < validation_count:
                    validationSet.append(image_array)
                    if "_l" in filename:
                        validationLabel.append(1)
                    elif "_r" in filename:
                        validationLabel.append(2)
                else :
                    testingSet.append(image_array)
                    if "_l" in filename:
                        testingLabel.append(1)
                    elif "_r" in filename:
                        testingLabel.append(2)
                total_count = total_count + 1
            if "4ch" in filename:
                    image_patches_4ch.append(image_array)
                    image_patches_4ch_label.append(1)


    #Building non-annuili patches from top of the image
    for left in range(0, 200, 20):
        for top in range(0, 70, 20):
            image_array = []
            non_annuili_cropped_image = crop_image(gray_image,left,top).load()
            for w in range(20):
                for h in range(20):
                    pixel = non_annuili_cropped_image[w,h]
                    image_array.append(pixel)
                #annuili_image_matrix[w,h] = pixel
            if "2ch" in filename:
                if total_count < training_count:
                    trainingSet.append(image_array)
                    trainingLabel.append(0)
                elif total_count < validation_count:
                    validationSet.append(image_array)
                    validationLabel.append(0)
                else :
                    testingSet.append(image_array)
                    testingLabel.append(0)
                total_count = total_count + 1
            if "4ch" in filename:
                image_patches_4ch.append(image_array)
                image_patches_4ch_label.append(0)


    #Bulding patches from centres of the image to build poslitive annuili
    for left in range(87,100,5):
        for top in range(87,100,5):

            annuili_cropped_image = crop_image(gray_image,left,top).load()
            image_array = []
            for w in range(20):
                for h in range(20):
                    pixel = annuili_cropped_image[w,h]
                    image_array.append(pixel)
                    #annuili_image_matrix[w,h] = pixel
            if "2ch" in filename:
                if total_count < training_count:
                    trainingSet.append(image_array)
                    if "_l" in filename:
                        trainingLabel.append(1)
                    elif "_r" in filename:
                        trainingLabel.append(2)
                elif total_count < validation_count:
                    validationSet.append(image_array)
                    if "_l" in filename:
                        validationLabel.append(1)
                    elif "_r" in filename:
                        validationLabel.append(2)
                else :
                    testingSet.append(image_array)
                    if "_l" in filename:
                        testingLabel.append(1)
                    elif "_r" in filename:
                        testingLabel.append(2)
                total_count = total_count + 1
            if "4ch" in filename:
                    image_patches_4ch.append(image_array)
                    image_patches_4ch_label.append(1)


    #Building non-annuili patches from bottom of the page
    for left in range(0, 200, 20):
        for top in range(110, 160, 20):
            image_array = []
            non_annuili_cropped_image = crop_image(gray_image,left,top).load()
            for w in range(20):
                for h in range(20):
                    pixel =  non_annuili_cropped_image[w,h]
                    image_array.append(pixel)
                #annuili_image_matrix[w,h] = pixel
                if "2ch" in filename:
                    if total_count < training_count:
                        trainingSet.append(image_array)
                        if "_l" in filename:
                            trainingLabel.append(1)
                        elif "_r" in filename:
                            trainingLabel.append(2)
                    elif total_count < validation_count:
                        validationSet.append(image_array)
                        if "_l" in filename:
                            validationLabel.append(1)
                        elif "_r" in filename:
                            validationLabel.append(2)
                    else :
                        testingSet.append(image_array)
                        if "_l" in filename:
                            testingLabel.append(1)
                        elif "_r" in filename:
                            testingLabel.append(2)
                    total_count = total_count + 1
            if "4ch" in filename:
                image_patches_4ch.append(image_array)
                image_patches_4ch_label.append(0)

    #Bulding patches from centres of the image to build positive annuili
    for left in range(86,95,5):
        for top in range(86,95,5):

            annuili_cropped_image = crop_image(gray_image,left,top).load()
            image_array = []
            for w in range(20):
                for h in range(20):
                    pixel = annuili_cropped_image[w,h]
                    image_array.append(pixel)
                    #annuili_image_matrix[w,h] = pixel
            if "2ch" in filename:
                if total_count < training_count:
                    trainingSet.append(image_array)
                    if "_l" in filename:
                        trainingLabel.append(1)
                    elif "_r" in filename:
                        trainingLabel.append(2)
                elif total_count < validation_count:
                    validationSet.append(image_array)
                    if "_l" in filename:
                        validationLabel.append(1)
                    elif "_r" in filename:
                        validationLabel.append(2)
                else :
                    testingSet.append(image_array)
                    if "_l" in filename:
                        testingLabel.append(1)
                    elif "_r" in filename:
                        testingLabel.append(2)
                total_count = total_count + 1
            if "4ch" in filename:
                image_patches_4ch.append(image_array)
                image_patches_4ch_label.append(1)


    #Building non-annuili patches from left of annuili
    for left in range(0, 80, 20):
        for top in range(90, 140, 20):
            image_array = []
            non_annuili_cropped_image = crop_image(gray_image,left,top).load()
            for w in range(20):
                for h in range(20):
                    pixel =  non_annuili_cropped_image[w,h]
                    image_array.append(pixel)
                #annuili_image_matrix[w,h] = pixel
            if "2ch" in filename:
                if total_count < training_count:
                    trainingSet.append(image_array)
                    trainingLabel.append(0)
                elif total_count < validation_count:
                    validationSet.append(image_array)
                    validationLabel.append(0)
                else :
                    testingSet.append(image_array)
                    testingLabel.append(0)
                total_count = total_count + 1
            if "4ch" in filename:
                image_patches_4ch.append(image_array)
                image_patches_4ch_label.append(0)

    #Bulding patches from centres of the image to build positive annuili
    for left in range(85,96,5):
        for top in range(85,96,5):

            annuili_cropped_image = crop_image(gray_image,left,top).load()
            image_array = []
            for w in range(20):
                for h in range(20):
                    pixel = annuili_cropped_image[w,h]
                    image_array.append(pixel)
                    #annuili_image_matrix[w,h] = pixel
            if "2ch" in filename:
                if total_count < training_count:
                    trainingSet.append(image_array)
                    if "_l" in filename:
                        trainingLabel.append(1)
                    elif "_r" in filename:
                        trainingLabel.append(2)
                elif total_count < validation_count:
                    validationSet.append(image_array)
                    if "_l" in filename:
                        validationLabel.append(1)
                    elif "_r" in filename:
                        validationLabel.append(2)
                else :
                    testingSet.append(image_array)
                    if "_l" in filename:
                        testingLabel.append(1)
                    elif "_r" in filename:
                        testingLabel.append(2)
                total_count = total_count + 1
            if "4ch" in filename:
                    image_patches_4ch.append(image_array)
                    image_patches_4ch_label.append(1)

    #Building non-annuili patches from right side of annuili
    for left in range(110, 190, 20):
        for top in range(90, 140, 20):
            image_array = []
            non_annuili_cropped_image = crop_image(gray_image,left,top).load()
            for w in range(20):
                for h in range(20):
                    pixel =  non_annuili_cropped_image[w,h]
                    image_array.append(pixel)
                #annuili_image_matrix[w,h] = pixel
            if "2ch" in filename:
                if total_count < training_count:
                    trainingSet.append(image_array)
                    trainingLabel.append(0)
                elif total_count < validation_count:
                    validationSet.append(image_array)
                    validationLabel.append(0)
                else :
                    testingSet.append(image_array)
                    testingLabel.append(0)
                total_count = total_count + 1
            if "4ch" in filename:
                image_patches_4ch.append(image_array)
                image_patches_4ch_label.append(0)

    trainingSet,trainingLabel = list_shuffle(trainingSet,trainingLabel)
    validationSet,validationLabel = list_shuffle(validationSet,validationLabel)
    testingSet,testingLabel = list_shuffle(testingSet,testingLabel)
    return trainingSet,trainingLabel,validationSet,validationLabel,testingSet,testingLabel,image_patches_4ch,image_patches_4ch_label,total_count
    #putting all the cropped image arrays vertically
    #patches = np.vstack(patches_as_arrays)





def crop_image(img,left,top):
    width = 20
    height = 20
    right = left + width
    bottom = top +  height
    box = (left,top,right,bottom)
    cropped_image = img.crop(box)
    return cropped_image

if __name__ == "__main__":
    make_annuili_data()
