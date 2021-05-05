# -*- coding: utf-8 -*-
"""
myModels.py

file contains numerous calleable functions from my main file

each function will accept an input and output (for shape data) and return a model

Created on Sun May  2 17:01:21 2021

@author: Daniel
"""

import matplotlib.pyplot as plt
import numpy as np

import pandas as pd
import glob

#get to building the actual model
import tensorflow as tf
from tensorflow import keras
from keras import models
from keras import layers

from keras.metrics import FalseNegatives, Precision, Recall, FalsePositives
import tensorflow
import tensorflow.keras.layers as L

from keras.layers import GlobalAveragePooling2D, Multiply, Dense
from keras import backend as K
from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose, Conv3D
from keras.layers import Activation, add, multiply, Lambda
from keras.layers import AveragePooling2D, average, UpSampling2D, Dropout
from keras.optimizers import Adam, SGD, RMSprop
from keras.initializers import glorot_normal, random_normal, random_uniform
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping
from keras import backend as K
from keras.layers.normalization import BatchNormalization 
from keras.applications import VGG19, densenet
from keras.models import load_model

import sklearn
from keras.utils import to_categorical


def shallowNet(input_, output_):
    WIDTH = input_.shape[1]
    HEIGHT = input_.shape[2]
    
    shallow = tf.keras.Sequential()
    shallow.add(L.Input(shape=(WIDTH, HEIGHT, 1)))
    shallow.add(L.Conv2D(16, kernel_size=(7,7), activation='relu', padding='valid'))
    shallow.add(L.MaxPooling2D())
    shallow.add(L.Flatten())
    shallow.add(L.Dense(100, activation='relu'))
    shallow.add(L.Dense(50, activation='relu'))
    shallow.add(L.Dense(output_.shape[1], activation='softmax'))
    
    return shallow

def UNet(input_, output_, kernel=(3,3)):
    WIDTH = input_.shape[1]
    HEIGHT = input_.shape[2]
    
    L1 = L.Input(shape=(WIDTH,HEIGHT,1))
    print('L1.shape:', L1.shape)
    conv1 = L.Conv2D(filters=64, kernel_size=kernel, activation='relu', padding='same')(L1)
    print('conv1.shape:', conv1.shape)
    branch1 = L.Conv2D(filters=64, kernel_size=kernel, activation='relu', padding='same')(conv1)
    print('branch1.shape:', branch1.shape)
    maxpool1 = L.MaxPooling2D(pool_size=(2,2))(branch1)
    print('maxpool1.shape:', maxpool1.shape)
    
    conv2 = L.Conv2D(filters=128, kernel_size=kernel, activation='relu', padding='same')(maxpool1)
    print('conv2.shape:', conv2.shape)
    branch2 = L.Conv2D(filters=128, kernel_size=kernel, activation='relu', padding='same')(conv2)
    print('branch2.shape:', branch2.shape)
    maxpool2 = L.MaxPooling2D(pool_size=(2,2))(branch2)
    print('maxpool2.shape:', maxpool2.shape)
    
    conv3 = L.Conv2D(filters=256, kernel_size=kernel, activation='relu', padding='same')(maxpool2)
    print('conv3.shape:', conv3.shape)
    branch3 = L.Conv2D(filters=256, kernel_size=kernel, activation='relu', padding='same')(conv3)
    print('branch3.shape:', branch3.shape)
    maxpool3 = L.MaxPooling2D(pool_size=(2,2))(branch3)
    print('maxpool3.shape:', maxpool3.shape)
    
    conv4 = L.Conv2D(filters=512, kernel_size=kernel, activation='relu', padding='same')(maxpool3)
    print('conv4.shape:', conv4.shape)
    branch4 = L.Conv2D(filters=512, kernel_size=kernel, activation='relu', padding='same')(conv4)
    print('branch4.shape:', branch4.shape)
    maxpool4 = L.MaxPooling2D(pool_size=(2,2))(branch4)
    
    conv5 = L.Conv2D(filters=1024, kernel_size=kernel, activation='relu', padding='same')(maxpool4)
    print('conv5.shape:', conv5.shape)
    conv6 = L.Conv2D(filters=1024, kernel_size=kernel, activation='relu', padding='same')(conv5)
    print('conv6.shape:', conv6.shape)
    up1 = L.UpSampling2D()(conv6)
    print('up1.shape:', up1.shape)
    
    concat4 = L.concatenate([branch4, up1]) #number beside branch and up should add up to 5
    print('concat4.shape:', concat4.shape)
    conv7 = L.Conv2D(filters=1024, kernel_size=kernel, activation='relu', padding='same')(concat4)
    print('conv7.shape:', conv7.shape)
    conv8 = L.Conv2D(filters=512, kernel_size=kernel, activation='relu', padding='same')(conv7)
    print('conv8.shape:', conv8.shape)
    up2 = L.UpSampling2D()(conv8)
    print('up2.shape', up2.shape)
    
    concat3 = L.concatenate([branch3, up2])
    print('concat3.shape:', concat3.shape)
    conv9 = L.Conv2D(filters=512, kernel_size=kernel, activation='relu', padding='same')(concat3)
    print('conv9.shape:', conv9.shape)
    conv10 = L.Conv2D(filters=256, kernel_size=kernel, activation='relu', padding='same')(conv9)
    print('conv10.shape:', conv10.shape)
    up3 = L.UpSampling2D()(conv10)
    print('up3.shape:', up3.shape)
    
    concat2 = L.concatenate([branch2, up3])
    print('concat2.shape:', concat2.shape)
    conv11 = L.Conv2D(filters=256, kernel_size=kernel, activation='relu', padding='same')(concat2)
    print('conv11.shape:', conv11.shape)
    conv12 = L.Conv2D(filters=128, kernel_size=kernel, activation='relu', padding='same')(conv11)
    print('conv12.shape:', conv12.shape)
    up4 = L.UpSampling2D()(conv12)
    print('up4.shape:', up4.shape)
    
    concat1 = L.concatenate([branch1, up4])
    print('concat1.shape:', concat1.shape)
    conv13 = L.Conv2D(filters=128, kernel_size=kernel, activation='relu', padding='same')(concat1)
    print('conv13.shape:', conv13.shape)
    conv14 = L.Conv2D(filters=64, kernel_size=kernel, activation='relu', padding='same')(conv13)
    print('conv14.shape:', conv14.shape)
    conv15 = L.Conv2D(filters=64, kernel_size=kernel, activation='relu', padding='same')(conv14)
    print('conv15.shape:', conv15.shape)
    output_layer = L.Conv2D(filters=1, kernel_size=(1,1), activation='softmax')(conv15)
    print('output_layer.shape:', output_layer.shape)
    flatten = L.Flatten()(output_layer)
    d1 = L.Dense(output_.shape[1], activation='softmax')(flatten)
    print('dense.shape:', d1.shape)
    
    unet = tf.keras.Model(inputs=L1, outputs=d1)
    
    return unet

def UNet_cross(input_, output_, kernel=(3,3)):
    WIDTH = input_.shape[1]
    HEIGHT = input_.shape[2]
    
    L1 = L.Input(shape=(WIDTH,HEIGHT,1))
    print('L1.shape:', L1.shape)
    conv1 = L.Conv2D(filters=64, kernel_size=kernel, activation='relu', padding='same')(L1)
    print('conv1.shape:', conv1.shape)
    branch1 = L.Conv2D(filters=64, kernel_size=kernel, activation='relu', padding='same')(conv1)
    print('branch1.shape:', branch1.shape)
    maxpool1 = L.MaxPooling2D(pool_size=(2,2))(branch1)
    print('maxpool1.shape:', maxpool1.shape)
    
    conv2 = L.Conv2D(filters=128, kernel_size=kernel, activation='relu', padding='same')(maxpool1)
    print('conv2.shape:', conv2.shape)
    branch2 = L.Conv2D(filters=128, kernel_size=kernel, activation='relu', padding='same')(conv2)
    print('branch2.shape:', branch2.shape)
    maxpool2 = L.MaxPooling2D(pool_size=(2,2))(branch2)
    print('maxpool2.shape:', maxpool2.shape)
    
    conv3 = L.Conv2D(filters=256, kernel_size=kernel, activation='relu', padding='same')(maxpool2)
    print('conv3.shape:', conv3.shape)
    branch3 = L.Conv2D(filters=256, kernel_size=kernel, activation='relu', padding='same')(conv3)
    print('branch3.shape:', branch3.shape)
    maxpool3 = L.MaxPooling2D(pool_size=(2,2))(branch3)
    print('maxpool3.shape:', maxpool3.shape)
    
    conv4 = L.Conv2D(filters=512, kernel_size=kernel, activation='relu', padding='same')(maxpool3)
    print('conv4.shape:', conv4.shape)
    branch4 = L.Conv2D(filters=512, kernel_size=kernel, activation='relu', padding='same')(conv4)
    print('branch4.shape:', branch4.shape)
    maxpool4 = L.MaxPooling2D(pool_size=(2,2))(branch4)
    
    conv5 = L.Conv2D(filters=1024, kernel_size=kernel, activation='relu', padding='same')(maxpool4)
    print('conv5.shape:', conv5.shape)
    conv6 = L.Conv2D(filters=1024, kernel_size=kernel, activation='relu', padding='same')(conv5)
    print('conv6.shape:', conv6.shape)
    up1 = L.UpSampling2D()(conv6)
    print('up1.shape:', up1.shape)
    
    concat4 = L.concatenate([branch4, up1]) #number beside branch and up should add up to 5
    print('concat4.shape:', concat4.shape)
    conv7 = L.Conv2D(filters=1024, kernel_size=kernel, activation='relu', padding='same')(concat4)
    print('conv7.shape:', conv7.shape)
    conv8 = L.Conv2D(filters=512, kernel_size=kernel, activation='relu', padding='same')(conv7)
    print('conv8.shape:', conv8.shape)
    up2 = L.UpSampling2D()(conv8)
    print('up2.shape', up2.shape)
    
    concat3 = L.concatenate([branch3, up2])
    print('concat3.shape:', concat3.shape)
    conv9 = L.Conv2D(filters=512, kernel_size=kernel, activation='relu', padding='same')(concat3)
    print('conv9.shape:', conv9.shape)
    conv10 = L.Conv2D(filters=256, kernel_size=kernel, activation='relu', padding='same')(conv9)
    print('conv10.shape:', conv10.shape)
    up3 = L.UpSampling2D()(conv10)
    print('up3.shape:', up3.shape)
    
    concat2 = L.concatenate([branch2, up3])
    print('concat2.shape:', concat2.shape)
    conv11 = L.Conv2D(filters=256, kernel_size=kernel, activation='relu', padding='same')(concat2)
    print('conv11.shape:', conv11.shape)
    conv12 = L.Conv2D(filters=128, kernel_size=kernel, activation='relu', padding='same')(conv11)
    print('conv12.shape:', conv12.shape)
    up4 = L.UpSampling2D()(conv12)
    print('up4.shape:', up4.shape)
    
    concat1 = L.concatenate([branch1, up4])
    print('concat1.shape:', concat1.shape)
    conv13 = L.Conv2D(filters=128, kernel_size=kernel, activation='relu', padding='same')(concat1)
    print('conv13.shape:', conv13.shape)
    conv14 = L.Conv2D(filters=64, kernel_size=kernel, activation='relu', padding='same')(conv13)
    print('conv14.shape:', conv14.shape)
    conv15 = L.Conv2D(filters=64, kernel_size=kernel, activation='relu', padding='same')(conv14)
    print('conv15.shape:', conv15.shape)
    seg_mask = L.Conv2D(filters=1, kernel_size=(1,1), activation='softmax')(conv15)
    print('seg_mask.shape:', seg_mask.shape)
    
    mult = L.multiply(seg_mask, L1)
    print('mult.shape:', mult.shape)
    
    flatten = L.Flatten()(mult)
    d1 = L.Dense(output_.shape[1], activation='softmax')(flatten)
    print('dense.shape:', d1.shape)
    
    unet_cross = tf.keras.Model(inputs=L1, outputs=d1)
    
    return unet_cross
    

def AlexNet(input_, output_):
    WIDTH = input_.shape[1]
    HEIGHT = input_.shape[2]
    
    alexnet = tf.keras.Sequential()
    alexnet.add(L.Input(shape=(WIDTH, HEIGHT, 1)))
    alexnet.add(L.Conv2D(96, kernel_size=(11,11), strides=4, activation='relu', padding='same'))
    alexnet.add(L.MaxPool2D(pool_size=(3,3), strides=2))
    alexnet.add(L.Conv2D(256, kernel_size=(5,5), padding='same', activation='relu'))
    alexnet.add(L.MaxPool2D(pool_size=(3,3), strides=2))
    alexnet.add(L.Conv2D(384, kernel_size=(3,3), padding='same', activation='relu'))
    alexnet.add(L.Conv2D(384, kernel_size=(3,3), padding='same', activation='relu'))
    alexnet.add(L.Conv2D(256, kernel_size=(3,3), padding='same', activation='relu'))
    alexnet.add(L.MaxPool2D(pool_size=(3,3), strides=2))
    alexnet.add(L.Flatten())
    alexnet.add(L.Dense(4096, activation='relu'))
    alexnet.add(L.Dense(4096, activation='relu'))
    alexnet.add(L.Dense(output_.shape[1], activation='softmax'))
    
    return alexnet