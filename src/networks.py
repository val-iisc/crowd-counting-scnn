# networks.py
# Defines all the networks of SCNN


import numpy as np
import theano
import theano.tensor as T
import lasagne

from lasagne.layers import InputLayer
from lasagne.layers import DenseLayer
from lasagne.layers import MaxPool2DLayer as PoolLayer
from lasagne.layers import Conv2DLayer as ConvLayer


def std_conv_layer(input, num_filters, filter_shape, pad = 'same'):
    return ConvLayer(input, num_filters, filter_shape,
                        pad = pad, flip_filters = False,
                        W = lasagne.init.Normal(std = 0.01, mean = 0.0))


# 9x9 Regressor (R1)
class shallow_net_9x9():
    
    def __init__(self, input = None, name = 'shallow_9x9'):
        if input is None:
            input = T.tensor4()
        self.input = input
        self.name = name
        net = {}
        net['input'] = InputLayer(shape = (None, 1, None, None),
                                  input_var = self.input)
        net['conv1_1'] = std_conv_layer(net['input'], 16, 9)
        net['pool1'] = PoolLayer(net['conv1_1'], 2)
        net['conv1_2'] = std_conv_layer(net['pool1'], 32, 7)
        net['pool2'] = PoolLayer(net['conv1_2'], 2)
        net['conv1_3'] = std_conv_layer(net['pool2'], 16, 7)
        net['conv1_4'] = std_conv_layer(net['conv1_3'], 8, 7)
        net['conv'] = std_conv_layer(net['conv1_4'], 1, 1)
        self.net = net
        self.output_layer = net['conv']
        self.train_output = lasagne.layers.get_output(self.output_layer)
        self.test_output = lasagne.layers.get_output(self.output_layer,
                                deterministic = True)
        self.parameters = lasagne.layers.get_all_params(self.output_layer,
                                                        trainable=True)


# 7x7 Regressor (R2)
class shallow_net_7x7():
    
    def __init__(self, input = None, name = 'shallow_7x7'):
        if input is None:
            input = T.tensor4()
        self.input = input
        self.name = name
        net = {}
        net['input'] = InputLayer(shape = (None, 1, None, None),
                                  input_var = self.input)
        net['conv2_1'] = std_conv_layer(net['input'], 20, 7)
        net['pool2_1'] = PoolLayer(net['conv2_1'], 2)
        net['conv2_2'] = std_conv_layer(net['pool2_1'], 40, 5)
        net['pool2_2'] = PoolLayer(net['conv2_2'], 2)
        net['conv2_3'] = std_conv_layer(net['pool2_2'], 20, 5)
        net['conv2_4'] = std_conv_layer(net['conv2_3'], 10, 5)
        net['conv'] = std_conv_layer(net['conv2_4'], 1, 1)
        self.net = net
        self.output_layer = net['conv']
        self.train_output = lasagne.layers.get_output(self.output_layer)
        self.test_output = lasagne.layers.get_output(self.output_layer,
                                deterministic = True)
        self.parameters = lasagne.layers.get_all_params(self.output_layer,
                                                        trainable=True)


# 5x5 Regressor (R3)
class shallow_net_5x5():
    
    def __init__(self, input = None, name = 'shallow_5x5'):
        if input is None:
            input = T.tensor4()
        self.input = input
        self.name = name
        net = {}
        net['input'] = InputLayer(shape = (None, 1, None, None),
                                  input_var = self.input)
        net['conv3_1'] = std_conv_layer(net['input'], 24, 5)
        net['pool3_1'] = PoolLayer(net['conv3_1'], 2)
        net['conv3_2'] = std_conv_layer(net['pool3_1'], 48, 3)
        net['pool3_2'] = PoolLayer(net['conv3_2'], 2)
        net['conv3_3'] = std_conv_layer(net['pool3_2'], 24, 3)
        net['conv3_4'] = std_conv_layer(net['conv3_3'], 12, 3)
        net['conv'] = std_conv_layer(net['conv3_4'], 1, 1)
        self.net = net
        self.output_layer = net['conv']
        self.train_output = lasagne.layers.get_output(self.output_layer)
        self.test_output = lasagne.layers.get_output(self.output_layer,
                                deterministic = True)
        self.parameters = lasagne.layers.get_all_params(self.output_layer,
                                                        trainable=True)


# Switch classifier
class deep_patch_classifier():
    
    def __init__(self, input = None, name = 'deep_patch_classifier'):
        if input is None:
            input = T.tensor4()
        self.input = input
        self.name = name
        rgb_means = T.as_tensor_variable(\
                                np.array([104.008, 116.669, 122.675],
                                dtype = theano.config.floatX));
        mean_sub_input = T.tile(self.input, [1, 3, 1, 1])
        mean_sub_input -= rgb_means.dimshuffle('x', 0, 'x', 'x')
        net = {}
        net['input'] = InputLayer(shape = (None, 3, None, None),
                                  input_var = mean_sub_input)
        net['conv1_1'] = ConvLayer(
                    net['input'], 64, 3, pad=1, flip_filters=False,
                    W = lasagne.init.Normal(std=0.01, mean=0.0))
        net['conv1_2'] = ConvLayer(
                    net['conv1_1'], 64, 3, pad=1, flip_filters=False,
                    W = lasagne.init.Normal(std=0.01, mean=0.0))
        net['pool1'] = PoolLayer(net['conv1_2'], 2)
        net['conv2_1'] = ConvLayer(
                    net['pool1'], 128, 3, pad=1, flip_filters=False,
                    W = lasagne.init.Normal(std=0.01, mean=0.0))
        net['conv2_2'] = ConvLayer(
                    net['conv2_1'], 128, 3, pad=1, flip_filters=False,
                    W = lasagne.init.Normal(std=0.01, mean=0.0))
        net['pool2'] = PoolLayer(net['conv2_2'], 2)
        net['conv3_1'] = ConvLayer(
                    net['pool2'], 256, 3, pad=1, flip_filters=False,
                    W = lasagne.init.Normal(std=0.01, mean=0.0))
        net['conv3_2'] = ConvLayer(
                    net['conv3_1'], 256, 3, pad=1, flip_filters=False,
                    W = lasagne.init.Normal(std=0.01, mean=0.0))
        net['conv3_3'] = ConvLayer(
                    net['conv3_2'], 256, 3, pad=1, flip_filters=False,
                    W = lasagne.init.Normal(std=0.01, mean=0.0))
        net['pool3'] = PoolLayer(net['conv3_3'], 2)
        net['conv4_1'] = ConvLayer(
                    net['pool3'], 512, 3, pad=1, flip_filters=False,
                    W = lasagne.init.Normal(std=0.01, mean=0.0))
        net['conv4_2'] = ConvLayer(
                    net['conv4_1'], 512, 3, pad=1, flip_filters=False,
                    W = lasagne.init.Normal(std=0.01, mean=0.0))
        net['conv4_3'] = ConvLayer(
                    net['conv4_2'], 512, 3, pad=1, flip_filters=False,
                    W = lasagne.init.Normal(std=0.01, mean=0.0))
        net['pool4'] = PoolLayer(net['conv4_3'], 2)
        net['conv5_1'] = ConvLayer(
                    net['pool4'], 512, 3, pad=1, flip_filters=False,
                    W = lasagne.init.Normal(std=0.01, mean=0.0))
        net['conv5_2'] = ConvLayer(
                    net['conv5_1'], 512, 3, pad=1, flip_filters=False,
                    W = lasagne.init.Normal(std=0.01, mean=0.0))
        net['conv5_3'] = ConvLayer(
                    net['conv5_2'], 512, 3, pad=1, flip_filters=False,
                    W = lasagne.init.Normal(std=0.01, mean=0.0))
        net['pool5'] = PoolLayer(net['conv5_3'], 2)
        net['gap'] = lasagne.layers.GlobalPoolLayer(net['pool5'])
        net['fc'] = DenseLayer(net['gap'], num_units=512)
        net['softmax'] = DenseLayer(net['fc'], num_units=3,
                            nonlinearity=lasagne.nonlinearities.softmax)
        self.output_layer = net['softmax']
        self.net = net
        self.train_output = lasagne.layers.get_output(net['softmax'])
        self.test_output = lasagne.layers.get_output(net['softmax'], deterministic=True)
        self.parameters = lasagne.layers.get_all_params(net['softmax'],
                                                        trainable=True)
    

# Create and compile theano graph for all the networks
def create_network_functions(networks):
    momentum = 0.9
    weight_decay = 0.0005
    run_funcs = []
    test_funcs = []
    train_funcs = []
    print 'Compiling theano functions...'
    for i, net in enumerate(networks):
        print ' > ' + net.name + '...'
        run_funcs.append(theano.function(inputs = [net.input],
                                          outputs = net.test_output))
        if i == 0:
            Y = T.lvector()
            train_loss = lasagne.objectives.categorical_crossentropy(net.train_output, Y).mean()
            test_loss = lasagne.objectives.categorical_crossentropy(net.test_output, Y).mean()
        else:
            Y = T.tensor4()
            train_loss = lasagne.objectives.squared_error(net.train_output, Y).mean()
            test_loss = lasagne.objectives.squared_error(net.test_output, Y).mean()
        test_funcs.append(theano.function(inputs = [net.input, Y],
                                          outputs = [test_loss, net.test_output]))
        learning_rate = T.scalar()
        updates = lasagne.updates.momentum(
                    train_loss, net.parameters, learning_rate = learning_rate, momentum=0.9)
        train_funcs.append(theano.function(
                        inputs = [net.input, Y, learning_rate],
                        outputs = train_loss,
                        updates = updates))
    print 'Done'
    return (train_funcs, test_funcs, run_funcs)


