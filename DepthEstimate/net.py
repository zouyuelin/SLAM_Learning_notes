import tensorflow as tf
import tensorflow.keras as K
import numpy as np

def model(dim_w,dim_h):
    First = K.layers.Input(shape=(dim_w,dim_h,3),name="input1")
    Second = K.layers.Input(shape=(dim_w,dim_h,3),name="input2")

    # x1 = K.layers.MaxPool2D(pool_size=(2,2),strides=2)(First)
    x1 = K.layers.Conv2D(128,kernel_size=(3,3), strides=2,padding='same')(First)
    x1 = K.layers.BatchNormalization()(x1)
    x1 = K.layers.LeakyReLU()(x1)
    x1 = K.layers.Conv2D(256,kernel_size=(3,3), strides=2,padding='same')(x1)
    x1 = K.layers.BatchNormalization()(x1)
    x1 = K.layers.ReLU()(x1)
    # x1 = ResNet34(x1,"x1")

    # x2 = K.layers.MaxPool2D(pool_size=(2,2),strides=2)(Second)
    x2 = K.layers.Conv2D(128,kernel_size=(3,3), strides=2,padding='same')(Second)
    x2 = K.layers.BatchNormalization()(x2)
    x2 = K.layers.LeakyReLU()(x2)
    x2 = K.layers.Conv2D(256,kernel_size=(3,3), strides=2,padding='same')(x2)
    x2 = K.layers.BatchNormalization()(x2)
    x2 = K.layers.ReLU()(x2)
    # x2 = ResNet34(x2,"x2")

    x = K.layers.concatenate([x1,x2])



    x = K.layers.Conv2DTranspose(256, kernel_size=(3,3), strides=2,padding='same')(x)
    x = K.layers.BatchNormalization()(x)
    x = K.layers.LeakyReLU()(x)

    x = K.layers.Conv2DTranspose(128, kernel_size=(3,3), strides=2,padding='same')(x)
    x = K.layers.BatchNormalization()(x)
    x = K.layers.LeakyReLU()(x)

    x = K.layers.Conv2D(32, kernel_size=(3,3), strides=1,padding='same')(x)
    x = K.layers.BatchNormalization()(x)
    x = K.layers.LeakyReLU()(x)

    x = K.layers.Conv2D(4, kernel_size=(3,3), strides=1,padding='same')(x)
    x = K.layers.BatchNormalization()(x)
    x = K.layers.LeakyReLU()(x)

    x = K.layers.Conv2D(1, kernel_size=(3,3), strides=1,padding='same',name='Output')(x)
    x = K.layers.ReLU()(x)

    poseModel = K.Model([First,Second],x)

    return poseModel