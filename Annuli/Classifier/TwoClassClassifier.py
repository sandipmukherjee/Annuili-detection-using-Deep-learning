#!/usr/bin/env python
# encoding: utf-8
import Image
import numpy as np

import ImageFilter
import ImageChops
import ImageOps
import scipy.ndimage as ndi
import scipy.misc
import theano
import theano.tensor as T
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv
import sys
import os
import cPickle

sys.path.append("..") # hack necessary to allow import from parent dir.

from NeuralNetwork.LogisticRegression import LogisticRegression
from NeuralNetwork.neuralnet import HiddenLayer
from NeuralNetwork.ConvolutionalNN import LeNetConvPoolLayer

from Preprocessing.preprocessing import preprocessing

def classify_lenet5(params, dataSet, nKerns=[20, 50], input_shape = 20):
    rng = np.random.RandomState(23455)
    test_set_x = dataSet
    # compute number of minibatches for training, validation and testing
    n_test_batches = test_set_x.get_value(borrow=True).shape[0]

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch
    x = T.matrix('x')   # the data is presented as rasterized images
    y = T.ivector('y')  # the labels are presented as 1D vector of
    # [int] labels

    ishape = (input_shape, input_shape)  # this is the size of images

    # Reshape matrix of rasterized images of shape (batch_size,28*28)
    # to a 4D tensor, compatible with our LeNetConvPoolLayer
    layer0_input = x.reshape((n_test_batches, 1, input_shape, input_shape))

    # Construct the first convolutional pooling layer:
    # filtering reduces the image size to (28-5+1,28-5+1)=(24,24)
    # maxpooling reduces this further to (24/2,24/2) = (12,12)
    # 4D output tensor is thus of shape (batch_size,nkerns[0],12,12)
    layer0_out_size = (input_shape - 5 + 1) / 2
    layer0 = LeNetConvPoolLayer(rng, input=layer0_input,
                                image_shape=(n_test_batches, 1, input_shape, input_shape),
                                filter_shape=(nKerns[0], 1, 5, 5), poolsize=(2, 2), W=params[0][0], b=params[0][1])

    # Construct the second convolutional pooling layer
    # filtering reduces the image size to (12-5+1,12-5+1)=(8,8)
    # maxpooling reduces this further to (8/2,8/2) = (4,4)
    # 4D output tensor is thus of shape (nkerns[0],nkerns[1],4,4)
    layer1_out_size = (layer0_out_size - 5 + 1) / 2
    layer1 = LeNetConvPoolLayer(rng, input=layer0.output,
                                image_shape=(n_test_batches, nKerns[0], layer0_out_size, layer0_out_size),
                                filter_shape=(nKerns[1], nKerns[0], 5, 5), poolsize=(2, 2), W=params[1][0], b=params[1][1])
    # the TanhLayer being fully-connected, it operates on 2D matrices of
    # shape (batch_size,num_pixels) (i.e matrix of rasterized images).
    # This will generate a matrix of shape (20,32*4*4) = (20,512)
    layer2_input = layer1.output.flatten(2)

    # construct a fully-connected sigmoidal layer
    layer2 = HiddenLayer(rng, input=layer2_input, input_dimensions=nKerns[1] * layer1_out_size * layer1_out_size,
                         output_dimensions=500, activation_function=T.tanh, Weight=params[2][0], bias=params[2][1])
    # classify the values of the fully-connected sigmoidal layer
    layer3 = LogisticRegression(input=layer2.output, input_dimensions=500, output_dimensions=2, params=params[3])

    # create a function to compute the mistakes that are made by the model
    test_model = theano.function([], outputs=layer3.getPrediction(),givens={  x: test_set_x})
    y = test_model()
    return y


def get_img_size(file_name):
    img = Image.open(file_name)
    return img.size

def main():
    img_names = []
    two_ch = "--2ch" in sys.argv
    four_ch = "--4ch" in sys.argv

    patch_size = 20
    if "--patch-size" in sys.argv:
        i = sys.argv.index("--patch-size") + 1
        if i < len(sys.argv) - 1:
            patch_size = int(sys.argv[i])

    print "Patch size: ", patch_size

    if not two_ch and not four_ch: # if no options are set, take whole dataset
        two_ch = True
        four_ch = True
    if two_ch:
        print(" 2 chamber classifier")
    if four_ch:
        print("4 chamber classifier")

    params = sys.argv[-2]
    paramsfile = open(params, 'rb')
    paramsdata = cPickle.load(paramsfile)

    data_dir = sys.argv[-1]
    if not data_dir[-1] == os.path.sep:
        data_dir = data_dir + os.path.sep

    preprocessing(data_dir)

    patch_offset = patch_size / 2  # the amount that the overlapping patches are offset from one to the next

    for dirname, dirnames, filenames in os.walk(data_dir):
        for filename in filenames:
            if filename.endswith(".jpg") or filename.endswith(".jpeg") or filename.endswith(".JPG"):
                    img_names.append(data_dir + filename)
    del filenames

    size = get_img_size(img_names[0])

    print("Generating positions")
    i_coords = [x * patch_offset for x in range(int((size[0] - patch_offset) / float(patch_offset)))]
    j_coords = [x * patch_offset for x in range(int((size[1] - patch_offset) / float(patch_offset)))]
    positions = [(i,j) for i in i_coords for j in j_coords]
    patch_count = len(positions)
    print patch_count
    print("Initializing data structures")
    img_count = len(img_names)
    for img_name in img_names:
        img = Image.open(img_name)
        all_samples = np.zeros((patch_count, patch_size * patch_size), dtype=np.uint8)
        if img.mode != "L":
            print("WARNING: Got image with mode %s instead of L! Are you using the cleaned dataset?" % img.mode)
        indx = 0
        for (i,j) in positions:
            annulus = img.crop((i,j,i+patch_size, j + patch_size))
            sample = np.hstack(np.asarray(annulus,dtype=np.uint8).flatten())
            all_samples[indx] = sample
            indx = indx + 1
        testing_data_inputs = theano.shared(np.asarray(all_samples, dtype=theano.config.floatX))
        prediction = classify_lenet5(paramsdata, testing_data_inputs, input_shape=patch_size)
        print prediction
        annulus_found = False
        count = 0
        for (i,j), k in zip(positions, prediction):
            if k == 1:
                count = count + 1
                print "Image %s, has annulus at the patch from position (%d, %d) to (%d, %d)"%(str(img_name), i, j, i + patch_size, j + patch_size)
                annulus_found = True
        print count
        if annulus_found == False:
            print "Image %s, has no annulus"%(str(img_name))


if __name__ == '__main__':
    #To run this file pass, --2ch or --4ch, --patch-size, pickled weights file, directory with jpeg images to be classified,
    main()

