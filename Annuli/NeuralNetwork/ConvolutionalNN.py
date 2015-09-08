__authors__ = 'anupamajha, Marcel Ruegenberg'
"""
Based on the LeNet5 code from the deeplearning.net tutorial.
"""
import cPickle
import gzip
import os
import sys
import time

import numpy
import numpy as np
import matplotlib.pyplot as plt

import theano
import theano.tensor as T
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv

from dataloading import load_data
from LogisticRegression import LogisticRegression
from neuralnet import HiddenLayer

class LeNetConvPoolLayer(object):
    """Pool Layer of a convolutional network """

    def __init__(self, rng, input, filter_shape, image_shape, poolsize=(2, 2), W=None, b=None):
        """
        Allocate a LeNetConvPoolLayer with shared variable internal parameters.

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dtensor4
        :param input: symbolic image tensor, of shape image_shape

        :type filter_shape: tuple or list of length 4
        :param filter_shape: (number of filters, num input feature maps,
                              filter height,filter width)

        :type image_shape: tuple or list of length 4
        :param image_shape: (batch size, num input feature maps,
                             image height, image width)

        :type poolsize: tuple or list of length 2
        :param poolsize: the downsampling (pooling) factor (#rows,#cols)
        """

        assert image_shape[1] == filter_shape[1]
        self.input = input

        # there are "num input feature maps * filter height * filter width"
        # inputs to each hidden unit
        fan_in = np.prod(filter_shape[1:])
        # each unit in the lower layer receives a gradient from:
        # "num output feature maps * filter height * filter width" /
        #   pooling size
        fan_out = (filter_shape[0] * np.prod(filter_shape[2:]) /
                   np.prod(poolsize))
        # initialize weights with random weights
        if W==None:
            W_bound = np.sqrt(6. / (fan_in + fan_out))
            self.W = theano.shared(np.asarray(
                rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
                dtype=theano.config.floatX),
                                   borrow=True)
        else:
            self.W=W

        # the bias is a 1D tensor -- one bias per output feature map
        if b==None:
            b_values = np.zeros((filter_shape[0],), dtype=theano.config.floatX)
            self.b = theano.shared(value=b_values, borrow=True)
        else:
            self.b=b

        # convolve input feature maps with filters
        conv_out = conv.conv2d(input=input, filters=self.W,
                               filter_shape=filter_shape, image_shape=image_shape)

        # downsample each feature map individually, using maxpooling
        pooled_out = downsample.max_pool_2d(input=conv_out,
                                            ds=poolsize, ignore_border=True)

        # add the bias term. Since the bias is a vector (1D array), we first
        # reshape it to a tensor of shape (1,n_filters,1,1). Each bias will
        # thus be broadcasted across mini-batches and feature map
        # width & height
        self.output = T.tanh(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))

        # store parameters of this layer
        self.params = [self.W, self.b]
        
        
def print_confusion_matrix(conf_matrix, class_labels):
    class_cnt = len(class_labels)
    line = "%12s\t" % " "
    for j in range(class_cnt):
        line = "%s%12s\t" % (line, class_labels[j])
    print(line)
    for i in range(class_cnt):
        line = "%12s\t" % (class_labels[i])
        for j in range(class_cnt):
            line = "%s%12d\t" % (line, conf_matrix[i,j])
        print(line)
            


def evaluate_lenet5(learning_rate=0.1, n_epochs=2, momentum=0.9,
                    dataset='../DataGeneration/tomtec2chamber.pkl',
                    nkerns=[20, 50], batch_size=500, confusion_error=False, default_error=True, use_rmsprop=True,
                    verbose=False, params_file=None):
    """ 
    :type learning_rate: float
    :param learning_rate: learning rate used (factor for the stochastic
                          gradient)

    :type n_epochs: int
    :param n_epochs: maximal number of epochs to run the optimizer

    :type dataset: string
    :param dataset: path to the dataset used for training / testing

    :type nkerns: list of ints
    :param nkerns: number of kernels on each layer
    """

    train_errors_list = []
    validate_errors_list = []
    test_errors_list = []
    confusion_train_errors_list = []
    confusion_validate_errors_list = []
    confusion_test_errors_list = []

    rng = numpy.random.RandomState(23455)

    datasets = load_data(dataset)

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]
    input_shape = datasets[3]
    class_labels = datasets[4]
    class_count = len(class_labels)
    
    print "Classes: \t", class_labels
    print "Patch size: \t", input_shape
    if use_rmsprop:
        print "Learning rate: \t%f (before adjustment for rmsprop)" % (learning_rate)
    else:
        print "Learning rate: \t%f" % (learning_rate)
    print "Momentum: \t", momentum
    
    if use_rmsprop:
        learning_rate = learning_rate * 0.01 # rmsprop needs much smaller learning rates

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0]
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]
    n_test_batches = test_set_x.get_value(borrow=True).shape[0]
    n_train_batches /= batch_size
    n_valid_batches /= batch_size
    n_test_batches /= batch_size

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch
    x = T.matrix('x')   # the data is presented as rasterized images
    y = T.ivector('y')  # the labels are presented as 1D vector of
                        # [int] labels

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print '... building the model'
    
    if params_file:
        params = None
        with open(params_file, 'rb') as f:
            params = cPickle.load(f)
        print "Using existing weights"
        if not params:
            print "Warning: Existing weights were non-existent"
        layer0_W = params[0][0]
        layer0_b = params[0][1]
        layer1_W = params[1][0]
        layer1_b = params[1][1]
        layer2_W = params[2][0]
        layer2_b = params[2][1]
        layer3_p = params[3]
    else:
        layer0_W = None
        layer0_b = None
        layer1_W = None
        layer1_b = None
        layer2_W = None
        layer2_b = None
        layer3_p = None

    # Reshape matrix of rasterized images of shape (batch_size,28*28)
    # to a 4D tensor, compatible with our LeNetConvPoolLayer
    layer0_input = x.reshape((batch_size, 1, input_shape, input_shape))

    # Construct the first convolutional pooling layer:
    # filtering reduces the image size to (28-5+1,28-5+1)=(24,24)
    # maxpooling reduces this further to (24/2,24/2) = (12,12)
    # 4D output tensor is thus of shape (batch_size,nkerns[0],12,12)
    layer0_out_size = (input_shape - 5 + 1) / 2
    layer0 = LeNetConvPoolLayer(rng, input=layer0_input,
                                image_shape=(batch_size, 1, input_shape, input_shape),
                                filter_shape=(nkerns[0], 1, 5, 5), poolsize=(2, 2), W=layer0_W, b=layer0_b)

    # Construct the second convolutional pooling layer
    # filtering reduces the image size to (12-5+1,12-5+1)=(8,8)
    # maxpooling reduces this further to (8/2,8/2) = (4,4)
    # 4D output tensor is thus of shape (nkerns[0],nkerns[1],4,4)
    layer1_out_size = (layer0_out_size - 5 + 1) / 2
    layer1 = LeNetConvPoolLayer(rng, input=layer0.output,
                                image_shape=(batch_size, nkerns[0], layer0_out_size, layer0_out_size),
                                filter_shape=(nkerns[1], nkerns[0], 5, 5), poolsize=(2, 2), W=layer1_W, b=layer1_b)

    # the TanhLayer being fully-connected, it operates on 2D matrices of
    # shape (batch_size,num_pixels) (i.e matrix of rasterized images).
    # This will generate a matrix of shape (20,32*4*4) = (20,512)
    layer2_input = layer1.output.flatten(2)

    # construct a fully-connected sigmoidal layer
    layer2 = HiddenLayer(rng, input=layer2_input, input_dimensions=nkerns[1] * layer1_out_size * layer1_out_size,
                         output_dimensions=500, activation_function=T.tanh, Weight=layer2_W, bias=layer2_b)

    # classify the values of the fully-connected sigmoidal layer
    layer3 = LogisticRegression(input=layer2.output, input_dimensions=500, output_dimensions=class_count, params=layer3_p)

    # the cost we minimize during training is the NLL of the model
    cost = layer3.loss_nll(y)
    
    # create a function to compute the mistakes that are made by the model
    train_errors = theano.function(inputs=[index],
                                   outputs=layer3.prediction_accuracy(y),
                                   givens={
                                       x: train_set_x[index * batch_size: (index + 1) * batch_size],
                                       y: train_set_y[index * batch_size: (index + 1) * batch_size]})

    test_model = theano.function([index], layer3.prediction_accuracy(y),
                                 givens={
                                     x: test_set_x[index * batch_size: (index + 1) * batch_size],
                                     y: test_set_y[index * batch_size: (index + 1) * batch_size]})

    validate_model = theano.function([index], layer3.prediction_accuracy(y),
                                     givens={
                                         x: valid_set_x[index * batch_size: (index + 1) * batch_size],
                                         y: valid_set_y[index * batch_size: (index + 1) * batch_size]})


    #######################Confusion matrix code######################################
    confusion_model_train = theano.function([index], layer3.getPrediction(),
                                            givens={
                                                x: train_set_x[index * batch_size: (index + 1) * batch_size]})
    confusion_model_validate = theano.function([index], layer3.getPrediction(),
                                               givens={
                                                   x: valid_set_x[index * batch_size: (index + 1) * batch_size]})
    confusion_model_test = theano.function([index], layer3.getPrediction(),
                                           givens={
                                               x: test_set_x[index * batch_size: (index + 1) * batch_size]})

    confusion_model_train_y = theano.function([index], y,
                                              givens={
                                                  y: train_set_y[index * batch_size: (index + 1) * batch_size]})

    confusion_model_validate_y = theano.function([index], y,
                                                 givens={
                                                     y: valid_set_y[index * batch_size: (index + 1) * batch_size]})

    confusion_model_test_y = theano.function([index], y,
                                             givens={
                                                 y: test_set_y[index * batch_size: (index + 1) * batch_size]})
    ###################################################################################


    # create a list of all model parameters to be fit by gradient descent
    params = layer3.params + layer2.params + layer1.params + layer0.params

    # create a list of gradients for all model parameters
    grads = T.grad(cost, params)

    # train_model is a function that updates the model parameters by
    # SGD Since this model has many parameters, it would be tedious to
    # manually create an update rule for each model parameter. We thus
    # create the updates list by automatically looping over all
    # (params[i],grads[i]) pairs.
    updates = []
    
    if use_rmsprop:
        prev_step = [theano.shared(value=numpy.zeros(params[i].get_value().shape, dtype=theano.config.floatX)) for i in xrange(len(params))]            
        mean_squ  = [theano.shared(value=numpy.zeros(params[i].get_value().shape, dtype=theano.config.floatX)) for i in xrange(len(params))]
        
        for param_i, grad_i, step_i, mean_squ_i in zip(params, grads, prev_step, mean_squ):
            # note: ideally, we'd use the full gradient at the first iteration (when mean_squ_i is usually 0)
            new_mean_squ_i = T.add(T.mul(mean_squ_i, 0.9), T.mul(T.square(grad_i), 0.1)) 
            new_grad_i = T.div_proxy(grad_i, T.add(T.sqrt(new_mean_squ_i), 1e-08))
            new_step_i = momentum * step_i - learning_rate * new_grad_i
    
            updates.append((mean_squ_i, new_mean_squ_i))
            updates.append((param_i, param_i + new_step_i))
            updates.append((step_i, new_step_i))
    else:
        for param_i, grad_i in zip(params, grads):
            updates.append((param_i, param_i - learning_rate * grad_i))
    
    train_model = theano.function([index], cost, updates=updates,
                                  givens={
                                      x: train_set_x[index * batch_size: (index + 1) * batch_size],
                                      y: train_set_y[index * batch_size: (index + 1) * batch_size]})
    ###############
    # TRAIN MODEL #
    ###############
    print '... training'
    # early-stopping parameters
    patience = 10000  # look as this many examples regardless
    patience_increase = 2  # wait this much longer when a new best is
                           # found
    improvement_threshold = 0.995  # a relative improvement of this much is
                                   # considered significant
    validation_frequency = min(n_train_batches, patience / 2)
                           # go through this many
                           # minibatches before checking the network
                           # on the validation set; in this case we
                           # check every epoch

    best_params = None
    best_validation_loss = numpy.inf
    best_iter = 0
    test_score = 0.
    ############confusion error################
    confusion_best_validation_loss = numpy.inf
    confusion_best_iter = 0
    confusion_test_score = 0.
    ###########################################
    start_time = time.clock()
    
    costs = []

    epoch = 0
    done_looping = False

    while (epoch < n_epochs) and (not done_looping):
        print epoch
        epoch = epoch + 1
        for minibatch_index in xrange(n_train_batches):
            # current_learning_rate = learning_rate * np.amax([0.005, (0.8**epoch)]) # decaying learning rate

            iter = (epoch - 1) * n_train_batches + minibatch_index
            cost_ij = train_model(minibatch_index)
            costs.append(cost_ij)
            if verbose or iter % 100 == 0:
                print 'training @ iter = %d, cost = %f' % (iter, cost_ij)

            if (iter + 1) % validation_frequency == 0:
                
                if confusion_error == True:
                    confusion_validation_losses = [confusion_model_validate(i) for i
                                                   in xrange(n_valid_batches)]
                    y = [confusion_model_validate_y(i) for i
                         in xrange(n_valid_batches)]
                    prediction = numpy.array(confusion_validation_losses).flatten()
                    y = numpy.array(y).flatten()
                    conf_matrix = numpy.bincount(class_count * (y) + (prediction), \
                                                 minlength=class_count * class_count).reshape(class_count, class_count)
                    confusion_loss = 0
                    confusion_loss_count = 0
                    for i in range(conf_matrix.shape[0]):
                        for j in range(conf_matrix.shape[0]):
                            if i != j:
                                confusion_loss = confusion_loss + conf_matrix[i][j]
                            confusion_loss_count = confusion_loss_count + conf_matrix[i][j]
                    confusion_validation_loss = float(confusion_loss) / float(confusion_loss_count)

                    print('Q2 error(Using confusion matrix): epoch %i, validation error %f %%' % \
                          (epoch, confusion_validation_loss * 100.))
                          
                    if confusion_validation_loss < confusion_best_validation_loss:
                        confusion_train_errors = [confusion_model_train(i) for i
                                                  in xrange(n_train_batches)]
                        y = [confusion_model_train_y(i) for i
                             in xrange(n_train_batches)]
                        prediction = numpy.array(confusion_train_errors).flatten()
                        y = numpy.array(y).flatten()
                        conf_matrix = numpy.bincount(class_count * (y) + (prediction), \
                                                     minlength=class_count * class_count).reshape(class_count, class_count)
                        confusion_loss = 0
                        confusion_loss_count = 0
                        for i in range(conf_matrix.shape[0]):
                            for j in range(conf_matrix.shape[0]):
                                if i != j:
                                    confusion_loss = confusion_loss + conf_matrix[i][j]
                                confusion_loss_count = confusion_loss_count + conf_matrix[i][j]
                        confusion_train_error = float(confusion_loss) / float(confusion_loss_count)
                        
                        print('Q2 error(Using confusion matrix): epoch %i, validation error %f %%' % \
                              (epoch, confusion_validation_loss * 100.))
                        

                        #improve patience if loss improvement is good enough
                        if confusion_validation_loss < confusion_best_validation_loss * improvement_threshold:
                            patience = max(patience, iter * patience_increase)

                        # save best validation score and iteration number
                        confusion_best_validation_loss = confusion_validation_loss
                        confusion_best_iter = iter
                        confusion_test_losses = [confusion_model_test(i) for i
                                                 in xrange(n_test_batches)]
                        y = [confusion_model_test_y(i) for i
                             in xrange(n_test_batches)]
                        prediction = numpy.array(confusion_test_losses).flatten()
                        y = numpy.array(y).flatten()

                        conf_matrix = numpy.bincount(class_count * (y) + (prediction), \
                                                     minlength=class_count * class_count).reshape(class_count, class_count)
                        confusion_loss = 0
                        confusion_loss_count = 0
                        for i in range(conf_matrix.shape[0]):
                            for j in range(conf_matrix.shape[0]):
                                if i != j:
                                    confusion_loss = confusion_loss + conf_matrix[i][j]
                                confusion_loss_count = confusion_loss_count + conf_matrix[i][j]
                        confusion_test_score = float(confusion_loss) / float(confusion_loss_count)

                        print(('Q2 Error(Using confusion matrix)epoch %i, test error of best ''model %f %%') % \
                              (epoch, confusion_test_score * 100.))

                        print "Confusion matrix: Test Set"
                        print_confusion_matrix(conf_matrix, class_labels)

                        if class_count == 2:
                            true_positive = float(conf_matrix[1][1]) / float(confusion_loss_count)
                            true_negative = float(conf_matrix[0][0]) / float(confusion_loss_count)
                            false_positive = float(conf_matrix[0][1]) / float(confusion_loss_count)
                            false_negative = float(conf_matrix[1][0]) / float(confusion_loss_count)
                            print "True Positive(Annuli) %f %%" % (true_positive * 100.0)
                            print "True Negative(Non Annuli) %f %%" % (true_negative * 100.0)
                            print "False Positive %f %%" % (false_positive * 100.0)
                            print "False Negative %f %%" % (false_negative * 100.0)

                        confusion_train_errors_list.append(confusion_train_error)
                        confusion_validate_errors_list.append(confusion_best_validation_loss)
                        confusion_test_errors_list.append(confusion_test_score)
                        confusion_best_params = [layer0.params, layer1.params, layer2.params, layer3.params]
                        
                        with open("confusion_weights.pkl", "wb") as f:
                            cPickle.dump(confusion_best_params, f, cPickle.HIGHEST_PROTOCOL)
                            
                        visualize_errors([(confusion_train_errors_list, "confusion_train"),
                                          (confusion_validate_errors_list, "confusion_validate"),
                                          (confusion_test_errors_list, "confusion_test")],
                                         costs=costs,
                                         integrated=True,
                                         show=False)
                        with open("confusion_errors.pkl", "wb") as f:
                            cPickle.dump((confusion_train_errors_list,confusion_validate_errors_list,confusion_test_errors_list), f, protocol=cPickle.HIGHEST_PROTOCOL)

                if default_error == True:
                    # compute zero-one loss on validation set
                    train_losses = [train_errors(i) for i
                                    in xrange(n_train_batches)]
                    this_train_loss = numpy.mean(train_losses)

                    validation_losses = [validate_model(i) for i
                                         in xrange(n_valid_batches)]
                    this_validation_loss = numpy.mean(validation_losses)

                    print('epoch %i, training error %f %%, validation error %f %%' % \
                          (epoch, this_train_loss * 100, this_validation_loss * 100.))

                    # if we got the best validation score until now
                    if this_validation_loss < best_validation_loss:

                        #improve patience if loss improvement is good enough
                        if this_validation_loss < best_validation_loss * improvement_threshold:
                            patience = max(patience, iter * patience_increase)

                        # save best validation score and iteration number
                        best_validation_loss = this_validation_loss
                        best_iter = iter

                        # test it on the test set
                        test_losses = [test_model(i) for i in xrange(n_test_batches)]
                        test_score = numpy.mean(test_losses)

                        train_errors_list.append(this_train_loss)
                        validate_errors_list.append(best_validation_loss)
                        test_errors_list.append(test_score)

                        print(('     epoch %i, minibatch %i/%i, test error of best '
                               'model %f %%') %
                              (epoch, minibatch_index + 1, n_train_batches,
                               test_score * 100.))
                        best_params = [layer0.params, layer1.params, layer2.params, layer3.params]
                        
                        with open("weights.pkl", "wb") as f:
                            cPickle.dump(best_params, f, protocol=cPickle.HIGHEST_PROTOCOL)

            if patience <= iter:
                done_looping = True
                break

    end_time = time.clock()

    if confusion_error == True:
        print('Optimization complete.')
        print('Q2 Error (using confusion matrix)Best validation score of %f %% with test performance %f %%' %
              (confusion_best_validation_loss * 100., confusion_test_score * 100.))
        visualize_errors([(confusion_train_errors_list, "confusion_train"),
                          (confusion_validate_errors_list, "confusion_validate"),
                          (confusion_test_errors_list, "confusion_test")])
        with open("confusion_errors.pkl", "wb") as f:
            cPickle.dump((confusion_train_errors_list,confusion_validate_errors_list,confusion_test_errors_list), f, protocol=cPickle.HIGHEST_PROTOCOL)

    if default_error == True:
        print('Optimization complete.')
        print('Best validation score of %f %% with test performance %f %%' %
              (best_validation_loss * 100., test_score * 100.))
        visualize_errors([(train_errors_list, "train"), 
                          (validate_errors_list, "validate"), 
                          (test_errors_list, "test")])

    print >> sys.stderr, ('The code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.))


def visualize_errors(errors_arrays, costs=None, show=True, integrated=False):
    if integrated:
        plt.figure(1)
        plt.subplot(211)
        colors = ['r','g','b','c','m','y']
        i = 0
        for errors_array,split in errors_arrays:
            fmt_str = "%s.-" % colors[i]
            i = i+1
            plt.plot(errors_array, fmt_str, label=split)
        plt.legend()
        
        if costs:
            plt.subplot(212)
            plt.plot(costs, 'r.-', label='Cost / Error')
            plt.legend()
            
        plt.savefig("errors.png")

        if show:
            plt.show()
    else:
        for errors_array,split in errors_arrays:
            x = []
            for i in xrange(len(errors_array)):
                x.append(i)
            plt.plot(x, errors_array)
            plt.savefig("error_" + split + ".png")
            if show and split == "test":
                plt.show()
                