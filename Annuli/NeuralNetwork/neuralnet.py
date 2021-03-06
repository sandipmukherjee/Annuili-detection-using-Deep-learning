#!/usr/bin/env python
# encoding: utf-8
"""
neuralnet.py

Created by Anupama Jha on 2013-06-20.
"""
import cPickle
import numpy
import numpy as np

import theano
import theano.tensor as T
import theano.tensor as Tensor
from dataloading import load_data
import matplotlib.pyplot

class OutputLayer(object):
    def __init__(self, input, input_dimensions, output_dimensions):
        self.Weight = theano.shared(value=numpy.zeros((input_dimensions, output_dimensions), dtype=theano.config.floatX))
        self.bias = theano.shared(value=numpy.zeros((output_dimensions,),dtype=theano.config.floatX))
        self.class_conditional_probability = T.nnet.softmax(Tensor.dot(input, self.Weight) + self.bias)
        self.class_prediction = Tensor.argmax(self.class_conditional_probability, axis=1)
        self.Norm_L1 = abs(self.Weight).sum()
        self.Norm_L2 = abs(self.Weight ** 2).sum()
        self.parameters = [self.Weight, self.bias]
        self.meanSquarePerWeight = theano.shared(value=numpy.zeros((input_dimensions, output_dimensions), dtype=theano.config.floatX))
        self.meanSquarePerBias = theano.shared(value=numpy.zeros((output_dimensions,), dtype=theano.config.floatX))
        self.meanSquare = [self.meanSquarePerWeight, self.meanSquarePerBias]
        self.params = [self.Weight, self.bias]
    def change_val(self, input):
        self.class_conditional_probability = T.nnet.softmax(Tensor.dot(input, self.Weight) + self.bias)
        self.class_prediction = Tensor.argmax(self.class_conditional_probability, axis=1)

    def loss_nll(self, y):
        return -Tensor.mean(Tensor.log(self.class_conditional_probability)[Tensor.arange(y.shape[0]), y])

    def prediction_accuracy(self, y):
        return Tensor.mean(Tensor.neq(self.class_prediction, y))

class HiddenLayer(object):
    def __init__(self, random_val, input, input_dimensions, output_dimensions, Weight=None, bias=None,
                 activation_function=T.tanh):
        self.input = input
        self.meanSquarePerWeight = theano.shared(value=np.zeros((input_dimensions, output_dimensions), dtype=theano.config.floatX))
        self.meanSquarePerBias = theano.shared(value=np.zeros((output_dimensions,), dtype=theano.config.floatX))
        if Weight is None:
            Weight_values = np.asarray(random_val.uniform(
                    low=-np.sqrt(6. / (input_dimensions + output_dimensions)),
                    high=np.sqrt(6. / (input_dimensions + output_dimensions)),
                    size=(input_dimensions, output_dimensions)), dtype=theano.config.floatX)
            if activation_function == T.nnet.sigmoid:
                Weight_values *= 4
            Weight = theano.shared(value=Weight_values, name='Weight')

        if bias is None:
            bias_values = np.zeros((output_dimensions,), dtype=theano.config.floatX)
            bias = theano.shared(value=bias_values, name='bias')

        self.Weight = Weight
        self.bias = bias

        self.params = [self.Weight, self.bias]
        linear_output = T.dot(input, self.Weight) + self.bias
        self.output = (linear_output if activation_function is None
                       else activation_function(linear_output))
        self.parameters = [self.Weight, self.bias]
        self.meanSquare = [self.meanSquarePerWeight, self.meanSquarePerBias]


class NeuralNet(object):
    def __init__(self, random_val, input, input_dimensions, hidden_dimensions, output_dimensions):
        self.hiddenLayer = HiddenLayer(random_val=random_val, input=input,
                                       input_dimensions=input_dimensions, output_dimensions=hidden_dimensions,
                                       activation_function=T.tanh)

        self.logisticRegressionLayer = OutputLayer(
            input=self.hiddenLayer.output,
            input_dimensions=hidden_dimensions,
            output_dimensions=output_dimensions)

        self.norm_L1 = abs(self.hiddenLayer.Weight).sum() \
                + abs(self.logisticRegressionLayer.Weight).sum()

        self.norm_L2 = (self.hiddenLayer.Weight ** 2).sum() \
                    + (self.logisticRegressionLayer.Weight ** 2).sum()

        self.loss_nll = self.logisticRegressionLayer.loss_nll
        self.prediction_accuracy = self.logisticRegressionLayer.prediction_accuracy
        self.parameters = self.hiddenLayer.parameters + self.logisticRegressionLayer.parameters
        self.meanSquare = self.hiddenLayer.meanSquare + self.logisticRegressionLayer.meanSquare


# neural network with mini stochastic gradient descent
def optimize_neuralnet_msgd(learning_rate=0.001, lambda_1=0.00, lambda_2=0.0001, maximum_epochs=1000,
             dataset='tomtec2chamber.pkl', minibatch=200, hidden_units=1000):
    train_errors_list = []
    validate_errors_list = []
    test_errors_list = []
    data = load_data(dataset)
    print "data unpickled"
    training_data_inputs, training_data_labels = data[0]
    validation_data_inputs, validation_data_labels = data[1]
    testing_data_inputs, testing_data_labels = data[2]
    inp_dimensions = data[3]
    class_labels = data[4]

    minibatches_training = training_data_inputs.get_value().shape[0] / minibatch
    minibatches_validation = validation_data_inputs.get_value().shape[0] / minibatch
    minibatches_testing = testing_data_inputs.get_value().shape[0] / minibatch

    index = Tensor.lscalar()
    inputs = Tensor.matrix('inputs')
    labels = Tensor.ivector('labels')
    random_val = numpy.random.RandomState()

    classifier = NeuralNet(random_val=random_val, input=inputs, input_dimensions=(inp_dimensions * inp_dimensions),
                     hidden_dimensions=hidden_units, output_dimensions=2)

    loss_nll = classifier.loss_nll(labels) + lambda_1 * classifier.norm_L1 + lambda_2 * classifier.norm_L2

    function_for_testing = theano.function(inputs=[index],
            outputs=classifier.prediction_accuracy(labels),
            givens={
                inputs: testing_data_inputs[index * minibatch:(index + 1) * minibatch],
                labels: testing_data_labels[index * minibatch:(index + 1) * minibatch]})

    function_for_validation = theano.function(inputs=[index],
            outputs=classifier.prediction_accuracy(labels),
            givens={
                inputs: validation_data_inputs[index * minibatch:(index + 1) * minibatch],
                labels: validation_data_labels[index * minibatch:(index + 1) * minibatch]})

    gradient_parameters = []
    for param in classifier.parameters:
        gparam = Tensor.grad(loss_nll, param)
        gradient_parameters.append(gparam)
    learning = []
    for parameters, gradient_parameters in zip(classifier.parameters, gradient_parameters):
        learning.append((parameters, parameters - learning_rate * gradient_parameters))

    function_for_training = theano.function(inputs=[index], outputs=loss_nll,
            updates=learning,
            givens={
                inputs: training_data_inputs[index * minibatch:(index + 1) * minibatch],
                labels: training_data_labels[index * minibatch:(index + 1) * minibatch]})

    function_for_train_errors = theano.function(inputs=[index],
            outputs=classifier.prediction_accuracy(labels),
            givens={
                inputs: training_data_inputs[index * minibatch:(index + 1) * minibatch],
                labels: training_data_labels[index * minibatch:(index + 1) * minibatch]})

    training_loops = 240000
    best_parameters = None
    best_validate_error = numpy.inf
    test_error = 0.
    iteration = 0
    print "training with minibatch stochastic gradient Descent and Neural Network"
    for current_epoch in xrange(maximum_epochs):
        for index in xrange(minibatches_training):
            nll = function_for_training(index)
            iteration = (current_epoch) * minibatches_training + index
            if (iteration + 1) % minibatches_training == 0:
                validation_errors = [function_for_validation(i) for i in xrange(minibatches_validation)]
                current_validation_error = numpy.mean(validation_errors)
		train_errors = [function_for_train_errors(i)for i in xrange(minibatches_training)]
                train_error = numpy.mean(train_errors)
		print('training current_epoch number = %i : training error = %f :validation error = %f' %(current_epoch, train_error * 100., current_validation_error * 100.))

                if current_validation_error < best_validate_error:
                    best_validate_error = current_validation_error

                    test_errors = [function_for_testing(i) for i in xrange(minibatches_testing)]
                    test_error = numpy.mean(test_errors)
                    train_errors_list.append(train_error)
                    validate_errors_list.append(current_validation_error)
                    test_errors_list.append(test_error)
                    best_parameters = classifier.parameters
                    print('In training epoch %i with validation error = %i and test error = %f' %(current_epoch, best_validate_error * 100, test_error * 100.))
        if iteration >= training_loops:
            break

    print('minibatch gradient descent optimization done with validation error = %i and test error = %f' %(best_validate_error * 100, test_error * 100.))
    return best_parameters, train_errors_list, validate_errors_list, test_errors_list


#neural network with rmsprop
def optimize_neuralnet_rmsprop(learning_rate=0.1, lambda_1=0.00, lambda_2=0.0001, maximum_epochs=50,
             dataset='tomtec2chamber.pkl', minibatch=200, hidden_units=1000):
    train_errors_list = []
    validate_errors_list = []
    test_errors_list = []
    data = load_data(dataset)
    print "data unpickled"
    training_data_inputs, training_data_labels = data[0]
    validation_data_inputs, validation_data_labels = data[1]
    testing_data_inputs, testing_data_labels = data[2]
    inp_dimensions = data[3]

    minibatches_training = training_data_inputs.get_value().shape[0] / minibatch
    minibatches_validation = validation_data_inputs.get_value().shape[0] / minibatch
    minibatches_testing = testing_data_inputs.get_value().shape[0] / minibatch

    index = Tensor.lscalar()
    inputs = Tensor.matrix('inputs')
    labels = Tensor.ivector('labels')
    random_val = numpy.random.RandomState()

    classifier = NeuralNet(random_val=random_val, input=inputs, input_dimensions=(inp_dimensions * inp_dimensions),
                     hidden_dimensions=hidden_units, output_dimensions=2)

    loss_nll = classifier.loss_nll(labels) + lambda_1 * classifier.norm_L1 + lambda_2 * classifier.norm_L2

    function_for_testing = theano.function(inputs=[index],
            outputs=classifier.prediction_accuracy(labels),
            givens={
                inputs: testing_data_inputs[index * minibatch:(index + 1) * minibatch],
                labels: testing_data_labels[index * minibatch:(index + 1) * minibatch]})

    function_for_validation = theano.function(inputs=[index],
            outputs=classifier.prediction_accuracy(labels),
            givens={
                inputs: validation_data_inputs[index * minibatch:(index + 1) * minibatch],
                labels: validation_data_labels[index * minibatch:(index + 1) * minibatch]})

    gradient_parameters = []
    for param, meanSquare in zip(classifier.parameters,classifier.meanSquare):
        gparam = Tensor.grad(loss_nll, param)
        meanSquare = Tensor.mul(meanSquare, 0.9) + Tensor.mul(Tensor.pow(gparam, 2), 0.1)
        gparam = Tensor.div_proxy(gparam, Tensor.add(Tensor.sqrt(meanSquare), 1e-8))
        gradient_parameters.append(gparam)

    learning = []
    for parameters, gradient_parameters in zip(classifier.parameters, gradient_parameters):
        learning.append((parameters, parameters - learning_rate * gradient_parameters))

    function_for_training = theano.function(inputs=[index], outputs=loss_nll,
            updates=learning,
            givens={
                inputs: training_data_inputs[index * minibatch:(index + 1) * minibatch],
                labels: training_data_labels[index * minibatch:(index + 1) * minibatch]})

    function_for_train_errors = theano.function(inputs=[index],
            outputs=classifier.prediction_accuracy(labels),
            givens={
                inputs: training_data_inputs[index * minibatch:(index + 1) * minibatch],
                labels: training_data_labels[index * minibatch:(index + 1) * minibatch]})

    print "training with rmsprop and Neural Network"
    training_loops = 15000
    best_parameters = None
    best_validate_error = numpy.inf
    test_error = 0.

    for current_epoch in xrange(maximum_epochs):
        for index in xrange(minibatches_training):
            nll = function_for_training(index)
            iteration = (current_epoch) * minibatches_training + index
            if (iteration + 1) % minibatches_training == 0:
                validation_errors = [function_for_validation(i) for i in xrange(minibatches_validation)]
                current_validation_error = numpy.mean(validation_errors)
		train_errors = [function_for_train_errors(i)for i in xrange(minibatches_training)]
                train_error = numpy.mean(train_errors)
		print('training current_epoch number = %i : training error = %f :validation error = %f' %(current_epoch, train_error * 100., current_validation_error * 100.))
                if current_validation_error < best_validate_error:
                    best_validate_error = current_validation_error

                    test_errors = [function_for_testing(i) for i in xrange(minibatches_testing)]
                    test_error = numpy.mean(test_errors)

                    train_errors_list.append(train_error)
                    validate_errors_list.append(current_validation_error)
                    test_errors_list.append(test_error)
                    best_parameters = classifier.parameters

                    print('current_epoch number = %i :validation error = %f :test error = %f' %(current_epoch, current_validation_error * 100., test_error * 100.))

            if iteration >= training_loops:
                    break
    print('rmsprop optimization done :validation error = %f : test error = %f' %(current_validation_error * 100., test_error * 100.))
    return best_parameters, train_errors_list, validate_errors_list, test_errors_list

def Visualize_errors(errors_array, split):
    x = []
    for i in xrange(len(errors_array)):
        x.append(i)
    matplotlib.pyplot.plot(x, errors_array)
    matplotlib.pyplot.savefig("error_" + split + ".png")
    if split == "test":
        matplotlib.pyplot.show()

if __name__ == '__main__':

    #uncomment the call here to run msgd or rmsprop neural network and pass the data set name "tomtec4chamber.pkl"
    # or tomtec2chamber.pkl
    #weights, train_errors, valid_errors, test_errors = optimize_neuralnet_msgd(dataset="../DataGeneration/tomtec4chamber.pkl")
    #weights, train_errors, valid_errors, test_errors = optimize_neuralnet_msgd(dataset="tomtec4chamber.pkl")
    weights, train_errors, valid_errors, test_errors = optimize_neuralnet_rmsprop(dataset="/home/anupamajha/TUMSecondSemester/LabProject/Annuli/DataGeneration/dataset2ch.pkl")
    #weights, train_errors, valid_errors, test_errors = optimize_neuralnet_rmsprop(dataset="tomtec4chamber.pkl")
    
    # weights, train_errors, valid_errors, test_errors = optimize_neuralnet_rmsprop(dataset="../DataGeneration/dataset.pkl"
    Visualize_errors(train_errors, "train")
    Visualize_errors(valid_errors, "validate")
    Visualize_errors(test_errors, "test")
    cPickle.dump(weights, open("weights_neuralnet.pkl", "ab"))
