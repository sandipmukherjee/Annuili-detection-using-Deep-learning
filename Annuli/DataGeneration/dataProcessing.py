#!/usr/bin/env python
# encoding: utf-8
"""
dataProcessing.py

Created by Anupama Jha on 2013-06-20.
"""

from PIL import Image
import cPickle
import sys

def decompressImage(prefix):
    type1 = "000"
    type2 = "00"
    type3 = "0"
    chamber2 = "2ch"
    chamber4 = "4ch"
    left = "l"
    right = "r"
    suffix = ".jpg"
    underscore = "_"
    offset = 90
    imageList = []
    imageList1 = []
    imageLabel = []
    imageLabel1 = []
    trainingSet = []
    trainingLabel = []
    validationSet = []
    validationLabel = []
    testingSet = []
    testingLabel = []
    size1 = 0
    size2 = 0

    print "start generating patches for 2 chamber images"
    for i in range(3627):
        j = i + 1
        #for the 2 chamber left images
        if j < 10:
            imagePath = prefix + type1 +str(j)+underscore+ chamber2+underscore + left + suffix
        elif j >= 10 and j < 100:
             imagePath = prefix + type2 +str(j)+underscore+ chamber2 +underscore+ left + suffix
        elif j >= 100 and j < 1000:
            imagePath = prefix + type3 +str(j)+underscore+ chamber2 +underscore+ left + suffix
        else:
            imagePath = prefix +str(j) +underscore+ chamber2 +underscore+ left + suffix
        try:
            im = Image.open(imagePath);
            pix = im.load()
            image_positive = []
            positive_count = 0
            size1+= 1

            #creating the annuli patch
            for i in range(20):
                for j in range(20):
                    image_positive.append(pix[offset + i, offset + j])
            imageLabel.append(1)
            imageList.append(image_positive)

            #creating non annuli patches by taking 20 x 20 patch after every 10 pixels.
            #e.g. first patch starts at (0, 0), second at (0, 10)
            for l in range(0, 180, 10):
                for k in range(0, 180, 10):
                    size1+= 1
                    positive_count += 1
                    image = []
                    if (k > 70 and k < 110) and (l > 70 and l < 110):
                        pass
                    else:
                        for i in range(20):
                            for j in range(20):
                                image.append(pix[l + i, k + j])
                        imageLabel.append(0)
                        imageList.append(image)

                    #attaching an annuli patch after every five non annuli patches so that we have more positive samples
                    if(positive_count % 5) == 0:
                        size1 += 1
                        imageList.append(image_positive)
                        imageLabel.append(1)
        except IOError:
            pass

        #for the 2 chamber right images
        if j < 10:
            imagePath = prefix + type1 +str(j)+underscore+ chamber2+underscore + right + suffix
        elif j >= 10 and j < 100:
             imagePath = prefix + type2 +str(j)+underscore+ chamber2 +underscore+ right + suffix
        elif j >= 100 and j < 1000:
            imagePath = prefix + type3 +str(j)+underscore+ chamber2 +underscore+ right + suffix
        else:
            imagePath = prefix +str(j) +underscore+ chamber2 +underscore+ right+ suffix
        try:
            im = Image.open(imagePath);
            pix = im.load()
            image_positive = []
            positive_count = 0
            size1+= 1

            #creating the annuli patch
            for i in range(20):
                for j in range(20):
                    image_positive.append(pix[offset + i, offset + j])

            imageLabel.append(1)
            imageList.append(image_positive)

            #creating non annuli patches
            for l in range(0, 180, 10):
                for k in range(0, 180, 10):
                    size1+= 1
                    positive_count += 1
                    image = []
                    if (k > 70 and k < 110) and (l > 70 and l < 110):
                        pass
                    else:
                        for i in range(20):
                            for j in range(20):
                                image.append(pix[l + i, k + j])
                        imageLabel.append(0)
                        imageList.append(image)
                    #attaching an annuli patch after every five non annuli patches so that we have more positive samples
                    if(positive_count % 5) == 0:
                        size1 += 1
                        imageList.append(image_positive)
                        imageLabel.append(1)
        except IOError:
            pass
    print "finished generating patches for 2 chamber images"
    print "start generating patches for 4 chamber images"
    for i in range(3627):
        j = i + 1

        #for 4 chamber left images
        if j < 10:
            imagePath = prefix + type1 +str(j)+underscore+ chamber4+underscore + left + suffix
        elif j >= 10 and j < 100:
             imagePath = prefix + type2 +str(j)+underscore+ chamber4 +underscore+ left + suffix
        elif j >= 100 and j < 1000:
            imagePath = prefix + type3 +str(j)+underscore+ chamber4 +underscore+ left + suffix
        else:
            imagePath = prefix +str(j) +underscore+ chamber4 +underscore+ left + suffix
        try:
            im = Image.open(imagePath);
            pix = im.load()
            size2+= 1
            image_positive = []
            positive_count = 0

            #generating annuli patch
            for i in range(20):
                for j in range(20):
                    image_positive.append(pix[offset + i, offset + j])
            imageLabel1.append(1)
            imageList1.append(image_positive)

            #Generating non annuli patches
            for l in range(0, 180, 10):
                for k in range(0, 180, 10):
                    size2+= 1
                    positive_count += 1
                    image = []
                    if (k > 70 and k < 110) and (l > 70 and l < 110):
                        pass
                    else:
                        for i in range(20):
                            for j in range(20):
                                image.append(pix[l + i, k + j])
                        imageLabel1.append(0)
                        imageList1.append(image)
                    #attaching an annuli patch after every five non annuli patches so that we have more positive samples
                    if (positive_count % 5) == 0:
                        size2+= 1
                        imageList1.append(image_positive)
                        imageLabel1.append(1)

        except IOError:
            pass

        #for 4 chamber right images
        if j < 10:
            imagePath = prefix + type1 +str(j)+underscore+ chamber4+underscore + right + suffix
        elif j >= 10 and j < 100:
             imagePath = prefix + type2 +str(j)+underscore+ chamber4 +underscore+ right + suffix
        elif j >= 100 and j < 1000:
            imagePath = prefix + type3 +str(j)+underscore+ chamber4 +underscore+ right + suffix
        else:
            imagePath = prefix +str(j) +underscore+ chamber4 +underscore+ right+ suffix
        try:
            im = Image.open(imagePath);
            pix = im.load()
            positive_count = 0
            image_positive = []
            size2 += 1
            #Generating annuli patch
            for i in range(20):
                for j in range(20):
                    image_positive.append(pix[offset + i, offset + j])

            imageLabel1.append(1)
            imageList1.append(image_positive)

            #Generating non annuli patches
            for l in range(0, 180, 10):
                for k in range(0, 180, 10):
                    size2+= 1
                    positive_count += 1
                    image = []
                    if (k > 70 and k < 110) and (l > 70 and l < 110):
                        pass
                    else:
                        for i in range(20):
                            for j in range(20):
                                image.append(pix[l + i, k + j])
                        imageLabel1.append(0)
                        imageList1.append(image)
                    #attaching an annuli patch after every five non annuli patches so that we have more positive samples
                    if (positive_count % 5) == 0:
                        size2+= 1
                        imageList1.append(image_positive)
                        imageLabel1.append(1)
        except IOError:
            pass
    print "finished generating patches for 4 chamber images"

    print "training(60%), validation(20%) and testing(20%) split for 2 chamber data"
    #Generating training(60%), validation(20%) and testing(20%) split for 2 chamber data.
    counter = 0
    for image, label in zip(imageList, imageLabel):
        counter += 1
        if counter < ((3 * size1 ) / 5):
            trainingSet.append(image)
            trainingLabel.append(label)
        elif counter < ((4 * size1) /5):
            validationSet.append(image)
            validationLabel.append(label)
        else:
            testingSet.append(image)
            testingLabel.append(label)

    data1 = [trainingSet,trainingLabel,validationSet,validationLabel,testingSet,testingLabel]
    cPickle.dump( data1, open( "tomtec2chamber.pkl", "ab" ) )
    print "Data for 2 chamber images pickled to file tomtec2chamber.pkl at the present directory"

    print "training(60%), validation(20%) and testing(20%) split for 4 chamber data"
    #Generating training(60%), validation(20%) and testing(20%) split for 4 chamber data.
    trainingSet = []
    trainingLabel = []
    validationSet = []
    validationLabel = []
    testingSet = []
    testingLabel = []
    counter = 0
    for image, label in zip(imageList1, imageLabel1):
        counter += 1
        if counter < ((3 * size2 ) / 5):
            trainingSet.append(image)
            trainingLabel.append(label)
        elif counter < ((4 * size2) /5):
            validationSet.append(image)
            validationLabel.append(label)
        else:
            testingSet.append(image)
            testingLabel.append(label)
    data2 = [trainingSet,trainingLabel,validationSet,validationLabel,testingSet,testingLabel]
    cPickle.dump( data2, open( "tomtec4chamber.pkl", "ab" ) )
    print "Data for 4 chamber images pickled to file tomtec4chamber.pkl at the present directory"


if __name__ == "__main__":
    prefix = sys.argv[1]
    decompressImage(prefix)