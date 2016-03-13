# VGG-16, 16-layer model from the paper#onvolutional Networks for Large-Scale Image Recognition"
# Original source: https://gist.github.com/ksimonyan/211839e770f7b538e2d8
#
# Lcense: non-commercial use only


# Download pretrained weights from: http://s3.amazonaws.com/lasagne/recipes/pretrained/imagenet/vgg16.pkl
    
from lasagne.layers import InputLayer
from lasagne.layers import DenseLayer
from lasagne.layers import NonlinearityLayer
from lasagne.layers import DropoutLayer
from lasagne.layers import Pool2DLayer as PoolLayer
from lasagne.layers.dnn import Conv2DDNNLayer as ConvLayer
from lasagne.nonlinearities import softmax
import pickle
from __future__ import print_function
import load_data, sys, os, time, theano, vgg16
import lasagne
import numpy as np
import theano.tensor as T
import pickle
import matplotlib.pyplot as plt
import matplotlib

DROPOUT = 0.5

def build_model(input_var, BATCH_SIZE=None):
    pretrained_weights = pickle.load(open( "vgg16.pkl", "rb" ) )
    w = pretrained_weights['param values']
    print("w[0].shape", w[0].shape)
    print("w[27].shape", w[27].shape)
    new_w = [np.stack((weights, weights), axis = 1) for weights in w[:26]]
    new_w.extend(w[26:])
    print("new[0].shape", new_w[0].shape)
    print("new[27].shape", new_w[27].shape)
    net = {}
    net['input'] = InputLayer(shape=(BATCH_SIZE, 6, 224, 224), input_var=input_var)
    net['conv1_1'] = ConvLayer(
            net['input'], 64, 6, pad=1, flip_filters=False,
            W = w[0], b = w[1])
    print "hi"
    net['conv1_2'] = ConvLayer(
            net['conv1_1'], 64, 6, pad=1, flip_filters=False,
            W = w[2], b = w[3])
    net['pool1'] = PoolLayer(net['conv1_2'], 2)
    net['conv2_1'] = ConvLayer(
            net['pool1'], 128, 6, pad=1, flip_filters=False,
            W = w[4], b = w[5])
    net['conv2_2'] = ConvLayer(
            net['conv2_1'], 128, 6, pad=1, flip_filters=False,
            W = w[6], b = w[7])
    net['pool2'] = PoolLayer(net['conv2_2'], 2)
    net['conv3_1'] = ConvLayer(
            net['pool2'], 256, 6, pad=1, flip_filters=False,
            W = w[8], b = w[9])
    net['conv3_2'] = ConvLayer(
            net['conv3_1'], 256, 6, pad=1, flip_filters=False,
            W = w[10], b = w[11])
    net['conv3_3'] = ConvLayer(
            net['conv3_2'], 256, 6, pad=1, flip_filters=False,
            W = w[12], b = w[13])
    net['pool3'] = PoolLayer(net['conv3_3'], 2)
    net['conv4_1'] = ConvLayer(
            net['pool3'], 512, 6, pad=1, flip_filters=False,
            W = w[14], b = w[15])
    net['conv4_2'] = ConvLayer(
            net['conv4_1'], 512, 6, pad=1, flip_filters=False,
            W = w[16], b = w[17])
    net['conv4_3'] = ConvLayer(
            net['conv4_2'], 512, 6, pad=1, flip_filters=False,
            W = w[18], b = w[19])
    net['pool4'] = PoolLayer(net['conv4_3'], 2)
    net['conv5_1'] = ConvLayer(
            net['pool4'], 512, 6, pad=1, flip_filters=False,
            W = w[20], b = w[21])
    net['conv5_2'] = ConvLayer(
            net['conv5_1'], 512, 6, pad=1, flip_filters=False,
            W = w[22], b = w[23])
    net['conv5_3'] = ConvLayer(
            net['conv5_2'], 512, 6, pad=1, flip_filters=False,
            W = w[24], b = w[25])
    net['pool5'] = PoolLayer(net['conv5_3'], 2)
    net['fc6'] = DenseLayer(net['pool5'], num_units=4096,
            W = w[26], b = w[27])
    net['fc6_dropout'] = DropoutLayer(net['fc6'], p=DROPOUT)
    net['fc7'] = DenseLayer(net['fc6_dropout'], num_units=4096,
            W = w[28], b = w[29])
    net['fc7_dropout'] = DropoutLayer(net['fc7'], p=DROPOUT)
    net['fc8'] = DenseLayer(
            net['fc7_dropout'], num_units=2, nonlinearity=None)
    net['prob'] = NonlinearityLayer(net['fc8'], softmax)
    
    return net#

def main():
    build_model
    input_var = T.tensor4('inputs')
    target_var = T.ivector('targets')
    net = vgg16.build_model(input_var, 10)

if __name__ == "__main__":  
    main()
