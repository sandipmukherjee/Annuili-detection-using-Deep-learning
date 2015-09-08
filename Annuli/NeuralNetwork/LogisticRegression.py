__author__ = 'anupamajha'
import cPickle
import gzip
import numpy as np
import theano
import theano.tensor as T

class LogisticRegression(object):
    def __init__(self, input, input_dimensions, output_dimensions, params=None):
        if params==None:
            self.Weight = theano.shared(value=np.zeros((input_dimensions, output_dimensions), dtype=theano.config.floatX))
            self.bias = theano.shared(value=np.zeros((output_dimensions,), dtype=theano.config.floatX))
        else:
            self.Weight = params[0]
            self.bias = params[1]
        self.class_conditional_probability = T.nnet.softmax(T.dot(input, self.Weight) + self.bias)
        self.class_prediction = T.argmax(self.class_conditional_probability, axis=1)
        self.Norm_L1 = abs(self.Weight).sum()
        self.Norm_L2 = abs(self.Weight ** 2).sum()
        self.params = [self.Weight, self.bias]

    def change_val(self, input):
        self.class_conditional_probability = T.nnet.softmax(T.dot(input, self.Weight) + self.bias)
        self.class_prediction = T.argmax(self.class_conditional_probability, axis=1)

    def loss_nll(self, y):
        return -T.mean(T.log(self.class_conditional_probability)[T.arange(y.shape[0]), y])

    def prediction_accuracy(self, y):
        return T.mean(T.neq(self.class_prediction, y))

    def getPrediction(self):
        return self.class_prediction