"""
dropout.py

Created by Sandip Mukherjee
"""

import numpy as np
import cPickle
import gzip
import os
import sys
import time

import theano
import theano.tensor as T
from theano.ifelse import ifelse
import theano.printing
import theano.tensor.shared_randomstreams
import numpy
import logging
import sys
import matplotlib.pyplot as plt
from dataloading import load_data

class LogisticRegression(object):
    """Multi-class Logistic Regression Class

    The logistic regression is fully described by a weight matrix :math:`W`
    and bias vector :math:`b`. Classification is done by projecting data
    points onto a set of hyperplanes, the distance to which is used to
    determine a class membership probability.
    """

    def __init__(self, input, n_in, n_out, W=None, b=None):
        """ Initialize the parameters of the logistic regression

        :type input: theano.tensor.TensorType
        :param input: symbolic variable that describes the input of the
                      architecture (one minibatch)

        :type n_in: int
        :param n_in: number of input units, the dimension of the space in
                     which the datapoints lie

        :type n_out: int
        :param n_out: number of output units, the dimension of the space in
                      which the labels lie

        """

        # initialize with 0 the weights W as a matrix of shape (n_in, n_out)
        if W is None:
            self.W = theano.shared(
                    value=np.zeros((n_in, n_out), dtype=theano.config.floatX),
                    name='W')
        else:
            self.W = W

        # initialize the baises b as a vector of n_out 0s
        if b is None:
            self.b = theano.shared(
                    value=np.zeros((n_out,), dtype=theano.config.floatX),
                    name='b')
        else:
            self.b = b

        # compute vector of class-membership probabilities in symbolic form
        self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)

        # compute prediction as class whose probability is maximal in
        # symbolic form
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)

        # parameters of the model
        self.params = [self.W, self.b]

    def negative_log_likelihood(self, y):
        """Return the mean of the negative log-likelihood of the prediction
        of this model under a given target distribution.

        .. math::

            \frac{1}{|\mathcal{D}|} \mathcal{L} (\theta=\{W,b\}, \mathcal{D}) =
            \frac{1}{|\mathcal{D}|} \sum_{i=0}^{|\mathcal{D}|} \log(P(Y=y^{(i)}|x^{(i)}, W,b)) \\
                \ell (\theta=\{W,b\}, \mathcal{D})

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label

        Note: we use the mean instead of the sum so that
              the learning rate is less dependent on the batch size
        """
        # y.shape[0] is (symbolically) the number of rows in y, i.e.,
        # number of examples (call it n) in the minibatch
        # T.arange(y.shape[0]) is a symbolic vector which will contain
        # [0,1,2,... n-1] T.log(self.p_y_given_x) is a matrix of
        # Log-Probabilities (call it LP) with one row per example and
        # one column per class LP[T.arange(y.shape[0]),y] is a vector
        # v containing [LP[0,y[0]], LP[1,y[1]], LP[2,y[2]], ...,
        # LP[n-1,y[n-1]]] and T.mean(LP[T.arange(y.shape[0]),y]) is
        # the mean (across minibatch examples) of the elements in v,
        # i.e., the mean log-likelihood across the minibatch.
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])

    def errors(self, y):
        """Return a float representing the number of errors in the minibatch ;
        zero one loss over the size of the minibatch

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label
        """


        if y.dtype.startswith('int'):
            # the T.neq operator returns a vector of 0s and 1s, where 1
            # represents a mistake in prediction
            return T.mean(T.neq(self.y_pred, y))
        else:
            raise NotImplementedError()

class HiddenLayer(object):
    def __init__(self, rng, input, n_in, n_out,
                 activation, W=None, b=None,
                 use_bias=False):

        self.input = input
        self.activation = activation

        if W is None:
            W_values = np.asarray(0.01 * rng.standard_normal(
                size=(n_in, n_out)), dtype=theano.config.floatX)
            W = theano.shared(value=W_values, name='W')
        
        if b is None:
            b_values = np.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b')

        self.W = W
        self.b = b

        if use_bias:
            lin_output = T.dot(input, self.W) + self.b
        else:
            lin_output = T.dot(input, self.W)

        self.output = (lin_output if activation is None else activation(lin_output))
    
        # parameters of the model
        if use_bias:
            self.params = [self.W, self.b]
        else:
            self.params = [self.W]


def _dropout_layer_with_probability(rng, layer_values, p):
    """p is the probablity of dropping a unit.
    """
    random = theano.tensor.shared_randomstreams.RandomStreams(
            rng.randint(999999))
    # p is prob of dropping. hence 1-p is the prob to keep
    #binomial samples n times with probability of success prob for each trial, return the number of successes.
    mask = random.binomial(n=1, p=1-p, size=layer_values.shape)
    # masking the layer values with the mask
    output = layer_values * mask
    return output

class DropoutHiddenLayer(HiddenLayer):
    #this is a sublass of HiddenLayer where output of the layer is dropped out with probability 0.5
    def __init__(self, rng, input, n_in, n_out,
                 activation, use_bias, W=None, b=None):
        super(DropoutHiddenLayer, self).__init__(
                rng=rng, input=input, n_in=n_in, n_out=n_out, W=W, b=b,
                activation=activation, use_bias=use_bias)

        self.output = _dropout_layer_with_probability(rng, self.output, p=0.5)


class MLP(object):
    """A multilayer perceptron with dropout

    """
    def __init__(self,rng,input,layer_sizes,use_bias=True):

        activation = T.tanh

        # This will combine consecutive pairs of layer in a list like [(input,hidden1),(hidden1,hidden2),(hidden2,output)]
        consecutive_layer_sizes = zip(layer_sizes, layer_sizes[1:])

        #list of dropout layers
        self.dropout_layers = []
        # as suggested in paper, dropout the input with prob 0.2
        next_dropout_layer_input = _dropout_layer_with_probability(rng, input, p=0.2)
        for n_in, n_out in consecutive_layer_sizes[:-1]:
            next_dropout_layer = DropoutHiddenLayer(rng=rng,
                    input=next_dropout_layer_input,
                    activation=activation,
                    n_in=n_in, n_out=n_out, use_bias=use_bias)
            self.dropout_layers.append(next_dropout_layer)
            next_dropout_layer_input = next_dropout_layer.output

        # Set up the output layer as Logistic Regression Layer
        n_in, n_out = consecutive_layer_sizes[-1]
        dropout_output_layer = LogisticRegression(
                input=next_dropout_layer_input,
                n_in=n_in, n_out=n_out)
        self.dropout_layers.append(dropout_output_layer)



        # Objective function is the negative log likelihood of the logistic regression layer i.e the last layer in list of layers
        self.dropout_negative_log_likelihood = self.dropout_layers[-1].negative_log_likelihood
        #error is also the error of the logistic regression layer i.e the last layer in list of layers
        self.dropout_errors = self.dropout_layers[-1].errors


        # combine  all the parameters .
        self.params = [ param for layer in self.dropout_layers for param in layer.params ]





def test_dropout():

    #parameters
    initial_learning_rate = 0.1
    #learning rate will be decayed by this value after each epoch	
    learning_rate_decay = 0.98
    n_epochs = 50
    batch_size = 100
    dataset='dataset-2ch-2class-60px-unbal_test.pkl'
    use_bias = True

    validation_errors_array = []
    train_errors_array = []


    datasets = load_data(dataset)
    #datasets = load_data(dataset)
    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]
    input_shape = datasets[3]
    class_labels = datasets[4]
    class_count = len(class_labels)


    #mention layer sizes here [input,hidden1,hidden2,..,output]
    layer_sizes = [ input_shape*input_shape,1200,class_count]

    print "Standard NN with dropout..."
    print "Hidden layer dropout with 0.5 probability and input layer with 0.2 Probability"
    print "Layers sizes : \t" , layer_sizes
    print "Classes: \t", class_labels
    print "Patch size: \t", input_shape

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] / batch_size
    n_test_batches = test_set_x.get_value(borrow=True).shape[0] / batch_size

    logging.basicConfig(filename='logs/2ch_2class_60x60_dropout_1hidden_1200_drop_batch100_lr0.1_full_2class_final.log',level=logging.DEBUG)
    ######################
    # BUILD ACTUAL MODEL #
    ######################

    print '... building the model'

    # allocate symbolic variables for the data
    index = T.lscalar()    # index to a [mini]batch
    epoch = T.scalar()
    x = T.matrix('x')  # the data is presented as rasterized images
    y = T.ivector('y')  # the labels are presented as 1D vector of
                        # [int] labels
    learning_rate = theano.shared(np.asarray(initial_learning_rate,
        dtype=theano.config.floatX))

    rng = np.random.RandomState(1234)

    # construct the MLP class
    classifier = MLP(rng=rng, input=x,
           layer_sizes=layer_sizes, use_bias=use_bias)

    # cost function.
    dropout_cost = classifier.dropout_negative_log_likelihood(y)

    # Compile theano function for testing.
    test_model = theano.function(inputs=[index],
            outputs=classifier.dropout_errors(y),
            givens={
                x: test_set_x[index * batch_size:(index + 1) * batch_size],
                y: test_set_y[index * batch_size:(index + 1) * batch_size]})

    # Compile theano function for validation.
    validate_model = theano.function(inputs=[index],
            outputs=classifier.dropout_errors(y),
            givens={
                x: valid_set_x[index * batch_size:(index + 1) * batch_size],
                y: valid_set_y[index * batch_size:(index + 1) * batch_size]})

    # Compute gradients of the model wrt parameters
    gparams = []
    for param in classifier.params:
        # Use the cost function here to train with dropout.
        gparam = T.grad(dropout_cost, param)
        gparams.append(gparam)

    # allocatition of  memory for momentum'd versions of the gradient
    gparams_mom = []
    for param in classifier.params:
        gparam_mom = theano.shared(np.zeros(param.get_value(borrow=True).shape,
            dtype=theano.config.floatX))
        gparams_mom.append(gparam_mom)

    # Compute momentum for the current epoch according to hintons paper
    p_t = ifelse(epoch < 500,0.5*(1. - epoch/500.) + 0.99*(epoch/500.),0.99)

    # Updation of the step direction using momentum
    updates = {}
    for gparam_mom, gparam in zip(gparams_mom, gparams):
        updates[gparam_mom] = p_t * gparam_mom + (1. - p_t) * gparam

    # updating of the parameter
    for param, gparam_mom in zip(classifier.params, gparams_mom):
        stepped_param = param - (1.-p_t) * learning_rate * gparam_mom
        updates[param] = stepped_param

    # Compile theano function for training.  This returns the training cost and
    # updates the model parameters.
    train_output = [dropout_cost,classifier.dropout_errors(y)]
    output = dropout_cost
    train_model = theano.function(inputs=[epoch, index], outputs=train_output,
            updates=updates,
            givens={
                x: train_set_x[index * batch_size:(index + 1) * batch_size],
                y: train_set_y[index * batch_size:(index + 1) * batch_size]})

    # Theano function to decay the learning rate,after each epoch and not after one mini batch
    decay_learning_rate = theano.function(inputs=[], outputs=learning_rate,
            updates={learning_rate: learning_rate * learning_rate_decay})

    ###############
    # TRAIN MODEL #
    ###############
    print '... training'
    logging.debug('training')
    best_params = None
    best_validation_errors = np.inf
    best_iter = 0
    test_score = 0.
    epoch_counter = 0
    start_time = time.clock()


    while epoch_counter < n_epochs:
        # Train this epoch
        epoch_counter = epoch_counter + 1
        train_losses = []
        for minibatch_index in xrange(n_train_batches):

            train_output = train_model(epoch_counter, minibatch_index)
            iteration = (epoch_counter - 1) * n_train_batches + minibatch_index
            train_losses.append(train_output[1])
            logging.debug(('training error @ iter = %i : %f')%(iteration,train_output[1]))


        #compute train loss for 1 epoch
        train_errors_array.append(numpy.mean(train_losses))
        # Compute loss on validation set
        validation_losses = [validate_model(i) for i in xrange(n_valid_batches)]
        this_validation_errors = np.mean(validation_losses)
        validation_errors_array.append(this_validation_errors)
        # Report and save progress.
        print (('epoch %i, test error %f %%')%(
                epoch_counter, this_validation_errors*100.))
        logging.debug(('epoch %i, test error %f %%')%(
                epoch_counter, this_validation_errors*100.))
        best_validation_errors = min(best_validation_errors,
                this_validation_errors)


        learning_rate = decay_learning_rate()

    errors_arrays = [(train_errors_array, "train"),
                          (validation_errors_array, "validate")]
    end_time = time.clock()
    print 'Visualizing the weights and plotting the error curves'

    plt.figure(1)
    plt.subplot(211)
    colors = ['r','g','b','c','m','y']
    i = 0
    for errors_array,split in errors_arrays:
        fmt_str = "%s.-" % colors[i]
        i = i+1
        plt.plot(errors_array, fmt_str, label=split)
    plt.legend()
    plt.savefig("plots/2class_errors.png")


    print(('Optimization complete. Best validation score of %f %% '
           'obtained at iteration %i') %
          (best_validation_errors * 100., best_iter))
    logging.debug(('Optimization complete. Best validation score of %f %% '
           'obtained at iteration %i') %
          (best_validation_errors * 100., best_iter))
    print >> sys.stderr, ('The code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.))


if __name__ == '__main__':
    test_dropout()

