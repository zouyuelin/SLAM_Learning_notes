import argparse
import tensorflow as tf
import tensorflow.keras as K
import numpy as np
import cv2 as cv
import os
import time
import random
from tensorflow.keras import optimizers
from tensorflow.keras import callbacks

class datasets:
    def __init__(self, datasetsPath:str):
        self.dataPath = datasetsPath
        self.dim_w = 512
        self.dim_h = 512
        self.epochs = 40
        self.batch_size = 8
        self.train_percent = 0.92
        self.learning_rate = 2e-4
        self.model_path = 'kerasTempModel/'
        self.posetxt = os.path.join(self.dataPath,'pose.txt') 

        self.GetTheImagesAndPose()
        self.buildTrainData()

    def GetTheImagesAndPose(self):
        self.poselist = []
        with open(self.posetxt,'r') as f:
            for line in f.readlines():
                line = line.strip()
                line = line.split(' ')
                line.remove(line[0])
                self.poselist.append(line)
                # im1 im2 tx,ty,tz,roll,pitch,yaw
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
        pose = tf.strings.to_number(index[2:8],tf.float32)
        return (img_ref,img_cur),(pose)

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
           .repeat(10)\
           .batch(self.batch_size) \
           .prefetch(tf.data.experimental.AUTOTUNE)#.cache() 
        self.testdata = tf.data.Dataset.from_tensor_slices(self.test_pose_list) \
           .map(self.load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE) \
           .shuffle(500)\
           .repeat(10)\
           .batch(self.batch_size) \
           .prefetch(tf.data.experimental.AUTOTUNE)

def model(dim_w,dim_h):
    First = K.layers.Input(shape=(dim_w,dim_h,3),name="input1")
    Second = K.layers.Input(shape=(dim_w,dim_h,3),name="input2")

    x1 = K.layers.MaxPool2D(pool_size=(2,2),strides=2)(First)
    x1 = K.layers.Conv2D(512,kernel_size=(3,3), strides=2,padding='same')(x1)
    x1 = K.layers.BatchNormalization()(x1)
    x1 = K.layers.ReLU()(x1)
    x1 = K.layers.Conv2D(256,kernel_size=(3,3), strides=2,padding='same')(x1)
    x1 = K.layers.BatchNormalization()(x1)
    x1 = K.layers.ReLU()(x1)
    x1 = K.layers.MaxPool2D(pool_size=(2,2),strides=2)(x1)

    x2 = K.layers.MaxPool2D(pool_size=(2,2),strides=2)(Second)
    x2 = K.layers.Conv2D(512,kernel_size=(3,3), strides=2,padding='same')(x2)
    x2 = K.layers.BatchNormalization()(x2)
    x2 = K.layers.ReLU()(x2)
    x2 = K.layers.Conv2D(256,kernel_size=(3,3), strides=2,padding='same')(x2)
    x2 = K.layers.BatchNormalization()(x2)
    x2 = K.layers.ReLU()(x2)
    x2 = K.layers.MaxPool2D(pool_size=(2,2),strides=2)(x2)

    x = K.layers.concatenate([x1,x2])
    x = K.layers.Conv2D(256,kernel_size=(3,3), strides=1,padding='same',
                        activation='relu')(x)
    x = K.layers.BatchNormalization()(x)
    x = K.layers.ReLU()(x)
    x = K.layers.MaxPool2D(pool_size=(2,2),strides=2)(x)
    x = K.layers.Conv2D(128,kernel_size=(3,3), strides=1,padding='same',
                        activation='relu')(x)
    x = K.layers.BatchNormalization()(x)
    x = K.layers.ReLU()(x)
    x = K.layers.Flatten()(x)
    x = K.layers.Dense(1024)(x)
    x = K.layers.Dense(6,name='Output')(x)
    poseModel = K.Model([First,Second],x)

    return poseModel

def loss_fn(y_true,y_pre):
    loss_value = K.backend.mean(K.backend.square(y_true-y_pre))
    return loss_value


class learningDecay(K.callbacks.Callback):
    def __init__(self,schedule=None,alpha=1,verbose=0):
        super().__init__()
        self.schedule = schedule
        self.verbose = verbose
        self.alpha = alpha
    def on_epoch_begin(self, epoch, logs=None):
        lr = float(K.backend.get_value(self.model.optimizer.lr))
        if self.schedule != None:
            lr = self.schedule(epoch,lr)
        else:
            if epoch != 0:
                lr = lr*self.alpha
        K.backend.set_value(self.model.optimizer.lr,K.backend.get_value(lr))
        if self.verbose > 0:
            print(f"Current learning rate is {lr}")

def scheduler(epoch, lr):
    if epoch < 10:
        return lr
    else:
        return lr * tf.math.exp(-0.1) 

class Posenet:
    def __init__(self,dataset:datasets):
        self.dataset = dataset
        self.build()

    def build(self):
        self.poseModel = model(self.dataset.dim_w,self.dataset.dim_h)
        self.poseModel.summary()
        self.optm = K.optimizers.RMSprop(1e-4,momentum=0.9) #,decay=1e-5/self.dataset.epochs
        self.decayCallback = learningDecay(schedule = None,alpha = 0.99,verbose = 1)
        decayCallbackScheduler = K.callbacks.LearningRateScheduler(scheduler)
        self.callbacks = [decayCallbackScheduler]

        try:
            print("************************loading the model weights***********************************")
            self.poseModel.load_weights("model.h5")
        except:
            pass

    def train_fit(self):
        self.poseModel.compile(optimizer=self.optm,loss=loss_fn,metrics=['accuracy'])
        self.poseModel.fit(self.dataset.traindata,
                            validation_data=self.dataset.testdata,
                            epochs=self.dataset.epochs,
                            callbacks=[self.decayCallback],
                            verbose=1)
    
    def train_gradient(self):
        for step in range(self.dataset.epochs):
            loss = 0
            val_loss = 0
            lr = float(self.optm.lr)
            tf.print(">>> [Epoch is %s/%s]"%(step,self.dataset.epochs))
            for (x1,x2),y in self.dataset.traindata:
                with tf.GradientTape() as tape:
                    prediction = self.poseModel([x1,x2])
                    # y = tf.cast(y,dtype=prediction.dtype)
                    loss = loss_fn(y,prediction)
                gradients = tape.gradient(loss,self.poseModel.trainable_variables)
                self.optm.apply_gradients(zip(gradients,self.poseModel.trainable_variables))
            # 测试
            for (x1,x2),y in self.dataset.testdata:
                prediction = self.poseModel([x1,x2])
                val_loss = loss_fn(y,prediction)
            tf.print("The loss is %s,the learning rate is : %s, test loss is %s]"%(np.array(loss),lr,val_loss))
            K.backend.set_value(self.optm.lr,K.backend.get_value(lr*0.99))

    def save_model(self):
        '''
        利用 save 函数来保存，可以保存为h5文件，也可以保存为文件夹的形式，推荐保存第二种，再使用tf2onnx转onnx
        >>> python -m tf2onnx.convert --saved-model kerasTempModel --output "model.onnx" --opset 14
        '''
        self.poseModel.save("model.h5")
        self.poseModel.save(self.dataset.model_path)
        # self.poseModel.save_weights("model.h5") #只保存权重，没有保存结构
        # tf.saved_model.save(self.poseModel,'tf2TempModel') #这种保存方式不再使用了

        
def test(dataset):
    im1 = cv.imread("images/0.jpg")
    im1 = cv.resize(im1,(512,512))
    im1 = np.array(im1,np.float).reshape((1,512,512,3))/255.0
    im2 = cv.imread("images/1.jpg")
    im2 = cv.resize(im2,(512,512))
    im2 = np.array(im2,np.float).reshape((1,512,512,3))/255.0
    posemodel = K.models.load_model(dataset.model_path,compile=False)
    pose = posemodel([im1,im2])
    print(np.array(pose))

if __name__ == "__main__":
    dataset = datasets("images")
    posenet = Posenet(dataset)
    posenet.train_fit()
    # posenet.train_gradient() #利用 apply_gradient的方式训练
    posenet.save_model()
    test(dataset)
    
