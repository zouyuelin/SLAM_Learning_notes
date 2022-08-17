import argparse
import imp
import tensorflow as tf
import tensorflow.keras as K
import numpy as np
import cv2 as cv
import os
import time
import sys
import random

class datasets:
    def __init__(self, datasetsPath:str):
        self.dataPath = datasetsPath
        self.dim_w = 512
        self.dim_h = 512
        self.epochs = 200
        self.batch_size = 4
        self.train_percent = 0.92
        self.scale = 256.0
        self.learning_rate = 2e-3
        self.model_path = 'kerasTempModel/'
        self.posetxt = os.path.join(self.dataPath,'img.txt') 

        self.GetTheImagesAndPose()
        self.buildTrainData()

    def GetTheImagesAndPose(self):
        self.poselist = []
        with open(self.posetxt,'r') as f:
            for line in f.readlines():
                line = line.strip()
                line = line.split(' ')
                # line.remove(line[0])
                self.poselist.append(line)

        #打乱数据集
        length = np.shape(self.poselist)[0]
        train_num =int(length * self.train_percent) 
        test_num = length - train_num
        randomPoses = np.array(random.sample(self.poselist,length)) #取出所有数据集
        self.train_pose_list = randomPoses[0:train_num,:]
        self.test_pose_list = randomPoses[train_num:length+1,:]
        print(f"The size of train pose list is : {np.shape(self.train_pose_list)[0]}")
        print(f"The size of test pose list is : {np.shape(self.test_pose_list)[0]}")

    def load_image(self,index:tf.Tensor):

        img_ref= img_ref = tf.io.read_file(index[0])
        img_ref = tf.image.decode_jpeg(img_ref) #此处为jpeg格式
        #img = tf.reshape(img,[self.dim,self.dim,3])
        img_ref = tf.image.resize(img_ref,(self.dim_w,self.dim_h))/255.0
        img_ref = tf.cast(img_ref,tf.float32)

        img_cur = img_cur = tf.io.read_file(index[1])
        img_cur = tf.image.decode_jpeg(img_cur) #此处为jpeg格式
        img_cur = tf.image.resize(img_cur,(self.dim_w,self.dim_h))/255.0
        #img = tf.reshape(img,[self.dim,self.dim,3])
        img_cur = tf.cast(img_cur,tf.float32)

        groudTruth = tf.io.read_file(index[2])
        groudTruth = tf.image.decode_png(groudTruth,channels=1,dtype=tf.uint16) #此处为jpeg格式
        groudTruth = tf.image.resize(groudTruth,(self.dim_w,self.dim_h))
        groudTruth = tf.cast(groudTruth,tf.float32)/self.scale

        return (img_ref,img_cur),(groudTruth)

    def buildTrainData(self):
        '''
        for example:\\
        >>> poses = dataset.y_train.take(20)\\
        >>> imgs = dataset.x1_train.take(40)\\
        >>> print(np.array(list(imgs.as_numpy_iterator()))[39]) \\
        >>> imgs = dataset.x2_train.take(40)\\
        >>> print(np.array(list(imgs.as_numpy_iterator()))[39]) \\
        >>> print(np.array(list(poses.as_numpy_iterator()))[19]) \\
        '''
        self.traindata = tf.data.Dataset.from_tensor_slices(self.train_pose_list) \
           .map(self.load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE) \
           .shuffle(500)\
           .repeat(5)\
           .batch(self.batch_size) \
           .prefetch(tf.data.experimental.AUTOTUNE)#.cache() 
        self.testdata = tf.data.Dataset.from_tensor_slices(self.test_pose_list) \
           .map(self.load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE) \
           .shuffle(500)\
           .repeat(5)\
           .batch(self.batch_size) \
           .prefetch(tf.data.experimental.AUTOTUNE)