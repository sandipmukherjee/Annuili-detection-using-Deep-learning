#!/usr/bin/env python
# encoding: utf-8
"""
dataloading.py

Created by Marcel Ruegenberg on 2013-07-21.
Copyright (c) 2013 Dustlab. All rights reserved.
"""

import sys
import os
import cPickle

import numpy
import theano


def normalize(data):
    """Convert data to have 0 mean and 1 standard deviation in each sample"""
    data = numpy.asmatrix(data)
    std_devs = numpy.std(data, axis=1)
    std_devs[std_devs == 0] = 1 # prevent div by 0
    return (data - numpy.mean(data, axis=1)) / std_devs

def get_data(raw_data, normalize_data=True):
    data = raw_data
    training_data_inputs = data[0]
    training_data_labels = data[1]
    validation_data_inputs = data[2]
    validation_data_labels = data[3]
    testing_data_inputs = data[4]
    testing_data_labels = data[5]
    print "data loading successful"
    
    if normalize_data:
        training_data_inputs   = normalize(training_data_inputs)
        validation_data_inputs = normalize(validation_data_inputs)
        testing_data_inputs    = normalize(testing_data_inputs)
        print "Preprocessing successful"

    training_data_inputs = theano.shared(numpy.asarray(training_data_inputs,dtype=theano.config.floatX))
    training_data_labels = theano.shared(numpy.asarray(training_data_labels,dtype='int32'))
    validation_data_inputs = theano.shared(numpy.asarray( validation_data_inputs,dtype=theano.config.floatX))
    validation_data_labels = theano.shared(numpy.asarray( validation_data_labels,dtype='int32'))
    testing_data_inputs = theano.shared(numpy.asarray(testing_data_inputs,dtype=theano.config.floatX))
    testing_data_labels = theano.shared(numpy.asarray(testing_data_labels,dtype='int32'))

    partitioned_data = ((training_data_inputs, training_data_labels), \
                        (validation_data_inputs, validation_data_labels), \
                        (testing_data_inputs, testing_data_labels), \
                        20,["no annulus","annulus"])
    return partitioned_data
    
def get_data_gen_datasets(raw_data,normalize_data=True):
    """
    Get data packed by generate_datasets.py
    """

    training_set,validation_set,test_set,class_labels = raw_data
    print "Loading data succeeded"

    if normalize_data:
        training_set_samples   = normalize(training_set[:,0:-1])
        validation_set_samples = normalize(validation_set[:,0:-1])
        testing_set_samples    = normalize(test_set[:,0:-1])

        print "Normalized data"

    # assume square patches
    inp_shape = int(numpy.sqrt(training_set_samples.shape[1]))

    training_set_samples   = theano.shared(training_set_samples.astype(theano.config.floatX))
    validation_set_samples = theano.shared(validation_set_samples.astype(theano.config.floatX))
    test_set_samples       = theano.shared(testing_set_samples.astype(theano.config.floatX))

    training_set_labels   = theano.shared(training_set[:,-1].astype('int32')) # FIXME: uint8 should be sufficient
    validation_set_labels = theano.shared(validation_set[:,-1].astype('int32'))
    test_set_labels       = theano.shared(test_set[:,-1].astype('int32'))

    print "Unpacked data"

    return ((training_set_samples, training_set_labels), \
            (validation_set_samples, validation_set_labels), \
            (test_set_samples, test_set_labels), \
            inp_shape,class_labels)
                
def load_data(path,normalize_data=True):
    with open(path, 'rb') as f:
        raw_data = cPickle.load(f)
        if len(raw_data) == 4: # automatically detect which data generation code was used
            return get_data_gen_datasets(raw_data,normalize_data)
        else:
            return get_data(raw_data,normalize_data)
        
        