import argparse
import tensorflow as tf
import tensorflow.keras as K
import numpy as np
import cv2 as cv
import os
import time
import sys
import random
from tensorflow.keras import optimizers
from tensorflow.keras import callbacks
from tensorflow.python.keras.saving.save import save_model

class datasets:
    def __init__(self, datasetsPath:str):
        self.dataPath = datasetsPath
        self.dim_w = 512
        self.dim_h = 512
        self.epochs = 200
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

#--------resnet 50-------     
def identity_block(X, f, filters, stage, block, namescope):
    """
    :param X: input tensor
    :param f: shape for conv2 filter
    :param filters: List, the number of filters in the CONV layers of the main path
    :param stage: integer,used to name the layers, depending on their position in the network
    :param block: name
    :return:
    """
    conv_name_base = 'res' + str(stage) + block + '_branch' + namescope
    bn_name_base = 'bn' + str(stage) + block + '_branch' + namescope

    F1, F2, F3 = filters

    X_shortcut = X

    # first part
    X = K.layers.Conv2D(filters=F1, kernel_size=(1, 1), strides=(1, 1), padding='valid',
               name=conv_name_base + '2a', kernel_initializer=K.initializers.glorot_uniform(seed=0))(X)
    X = K.layers.BatchNormalization(axis=3, name=bn_name_base + '2a')(X)
    X = K.layers.Activation('relu')(X)

    # second part
    X = K.layers.Conv2D(filters=F2, kernel_size=(f, f), strides=(1, 1), padding='same',
               name=conv_name_base + '2b', kernel_initializer=K.initializers.glorot_uniform(seed=0))(X)
    X = K.layers.BatchNormalization(axis=3, name=bn_name_base + '2b')(X)
    X = K.layers.Activation('relu')(X)

    # third part
    X = K.layers.Conv2D(filters=F3, kernel_size=(1, 1), strides=(1, 1), padding='valid',
               name=conv_name_base + '2c', kernel_initializer=K.initializers.glorot_uniform(seed=0))(X)
    X = K.layers.BatchNormalization(axis=3, name=bn_name_base + '2c')(X)

    # combine the shortcut with the main path
    X = K.layers.add([X, X_shortcut])
    X = K.layers.Activation('relu')(X)

    return X

def convolutional_block(X, f, filters, stage, block, namescope, s=2):
    """
    :param X: input tensor
    :param f: shape of the second conv
    :param filters: list, the number of filters in the CONV layers of the main path
    :param stage: Integer, used to name the layers, depending on their position in the network
    :param block: same to the parameter stage
    :param s: Integer, stride
    :return:
    """
    conv_name_base = 'res' + str(stage) + block + '_branch'+namescope
    bn_name_base = 'bn' + str(stage) + block + '_branch'+namescope

    # Retrieve Filters
    F1, F2, F3 = filters

    # Save the input value
    X_shortcut = X

    # first part
    X = K.layers.Conv2D(filters=F1, kernel_size=(1, 1), strides=(s, s), padding='valid',
               name=conv_name_base + '2a', kernel_initializer=K.initializers.glorot_uniform(seed=0))(X)
    X = K.layers.BatchNormalization(axis=3, name=bn_name_base + '2a')(X)
    X = K.layers.Activation('relu')(X)

    # second part
    X = K.layers.Conv2D(filters=F2, kernel_size=(f, f), strides=(1, 1), padding='same',
               name=conv_name_base + '2b', kernel_initializer=K.initializers.glorot_uniform(seed=0))(X)
    X = K.layers.BatchNormalization(axis=3, name=bn_name_base + '2b')(X)
    X = K.layers.Activation('relu')(X)

    # third part
    X = K.layers.Conv2D(filters=F3, kernel_size=(1, 1), strides=(1, 1), padding='valid',
               name=conv_name_base + '2c', kernel_initializer=K.initializers.glorot_uniform(seed=0))(X)
    X = K.layers.BatchNormalization(axis=3, name=bn_name_base + '2c')(X)

    # shortcut path
    X_shortcut = K.layers.Conv2D(filters=F3, kernel_size=(1, 1), strides=(s, s), padding='valid',
                        name=conv_name_base + '1', kernel_initializer=K.initializers.glorot_uniform(seed=0))(X_shortcut)
    X_shortcut = K.layers.BatchNormalization(axis=3, name=bn_name_base + '1')(X_shortcut)

    X = K.layers.add([X, X_shortcut])
    X = K.layers.Activation('relu')(X)
    return X

def ResNet50(X_input,namescope="first"):
    X = K.layers.ZeroPadding2D((3, 3))(X_input)
    """
    阶段1：
    2D卷积具有64个形状为（7,7）的滤波器，并使用（2,2）步幅，名称是“conv1”。
    BatchNorm应用于输入的通道轴。
    MaxPooling使用（3,3）窗口和（2,2）步幅。
    """
    X = K.layers.Conv2D(filters=64, kernel_size=(7, 7), strides=(2, 2), name='conv1_'+namescope, kernel_initializer=K.initializers.glorot_uniform(seed=0))(X)
    X = K.layers.BatchNormalization(axis=3, name='bn_conv1'+namescope)(X)
    X = K.layers.Activation('relu')(X)
    X = K.layers.MaxPooling2D((3, 3), strides=(2, 2))(X)
    """
    阶段2：
    卷积块使用三组大小为[64,64,256]的滤波器，“f”为3，“s”为1，块为“a”。
    2个标识块使用三组大小为[64,64,256]的滤波器，“f”为3，块为“b”和“c”。
    """
    X = convolutional_block(X, f=3, filters=[64, 64, 256], stage=2, block='a',namescope=namescope, s=1)
    X = identity_block(X, f=3, filters=[64, 64, 256], stage=2, namescope=namescope, block='b')
    X = identity_block(X, f=3, filters=[64, 64, 256], stage=2, namescope=namescope, block='c')
    """
    阶段3：
    卷积块使用三组大小为[128,128,512]的滤波器，“f”为3，“s”为2，块为“a”。
    3个标识块使用三组大小为[128,128,512]的滤波器，“f”为3，块为“b”，“c”和“d”。
    """
    X = convolutional_block(X, f=3, filters=[128, 128, 512], stage=3, block='a',namescope=namescope, s=2)
    X = identity_block(X, f=3, filters=[128, 128, 512], stage=3, namescope=namescope, block='b')
    X = identity_block(X, f=3, filters=[128, 128, 512], stage=3, namescope=namescope, block='c')
    X = identity_block(X, f=3, filters=[128, 128, 512], stage=3, namescope=namescope, block='d')
    """
    阶段4：
    卷积块使用三组大小为[256、256、1024]的滤波器，“f”为3，“s”为2，块为“a”。
    5个标识块使用三组大小为[256、256、1024]的滤波器，“f”为3，块为“b”，“c”，“d”，“e”和“f”。
    """
    X = convolutional_block(X, f=3, filters=[256, 256, 1024], stage=4, block='a',namescope=namescope, s=2)
    X = identity_block(X, f=3, filters=[256, 256, 1024], stage=4, namescope=namescope, block='b')
    X = identity_block(X, f=3, filters=[256, 256, 1024], stage=4, namescope=namescope, block='c')
    X = identity_block(X, f=3, filters=[256, 256, 1024], stage=4, namescope=namescope, block='d')
    X = identity_block(X, f=3, filters=[256, 256, 1024], stage=4, namescope=namescope, block='e')
    X = identity_block(X, f=3, filters=[256, 256, 1024], stage=4, namescope=namescope, block='f')
    """
    阶段5：
    卷积块使用三组大小为[512、512、2048]的滤波器，“f”为3，“s”为2，块为“a”。
    2个标识块使用三组大小为[256、256、2048]的滤波器，“f”为3，块为“b”和“c”。
    """
    X = convolutional_block(X, f=3, filters=[512, 512, 2048], stage=5, block='a',namescope=namescope, s=2)
    X = identity_block(X, f=3, filters=[256, 256, 2048], stage=5, namescope=namescope, block='b')
    X = identity_block(X, f=3, filters=[256, 256, 2048], stage=5, namescope=namescope, block='c')

    X = K.layers.AveragePooling2D(pool_size=(2, 2))(X)
    return X


#-------resnet 34-------------
def conv_block(inputs, 
        neuron_num, 
        kernel_size,  
        use_bias, 
        padding= 'same',
        strides= (1, 1),
        with_conv_short_cut = False):
    conv1 = K.layers.Conv2D(
        neuron_num,
        kernel_size = kernel_size,
        activation= 'relu',
        strides= strides,
        use_bias= use_bias,
        padding= padding
    )(inputs)
    conv1 = K.layers.BatchNormalization(axis = 1)(conv1)

    conv2 = K.layers.Conv2D(
        neuron_num,
        kernel_size= kernel_size,
        activation= 'relu',
        use_bias= use_bias,
        padding= padding)(conv1)
    conv2 = K.layers.BatchNormalization(axis = 1)(conv2)

    if with_conv_short_cut:
        inputs = K.layers.Conv2D(
            neuron_num, 
            kernel_size= kernel_size,
            strides= strides,
            use_bias= use_bias,
            padding= padding
            )(inputs)
        return K.layers.add([inputs, conv2])

    else:
        return K.layers.add([inputs, conv2])

def ResNet34(inputs,namescope = ""):
    x = K.layers.ZeroPadding2D((3, 3))(inputs)

    # Define the converlutional block 1
    x = K.layers.Conv2D(64, kernel_size= (7, 7), strides= (2, 2), padding= 'valid')(x)
    x = K.layers.BatchNormalization(axis= 1)(x)
    x = K.layers.MaxPooling2D(pool_size= (3, 3), strides= (2, 2), padding= 'same')(x)

    # Define the converlutional block 2
    x = conv_block(x, neuron_num= 64, kernel_size= (3, 3), use_bias= True)
    x = conv_block(x, neuron_num= 64, kernel_size= (3, 3), use_bias= True)
    x = conv_block(x, neuron_num= 64, kernel_size= (3, 3), use_bias= True)

    # Define the converlutional block 3
    x = conv_block(x, neuron_num= 128, kernel_size= (3, 3), use_bias= True, strides= (2, 2), with_conv_short_cut= True)
    x = conv_block(x, neuron_num= 128, kernel_size= (3, 3), use_bias= True)
    x = conv_block(x, neuron_num= 128, kernel_size= (3, 3), use_bias= True)

    # Define the converlutional block 4
    x = conv_block(x, neuron_num= 256, kernel_size= (3, 3), use_bias= True, strides= (2, 2), with_conv_short_cut= True)
    x = conv_block(x, neuron_num= 256, kernel_size= (3, 3), use_bias= True)
    x = conv_block(x, neuron_num= 256, kernel_size= (3, 3), use_bias= True)
    x = conv_block(x, neuron_num= 256, kernel_size= (3, 3), use_bias= True)
    x = conv_block(x, neuron_num= 256, kernel_size= (3, 3), use_bias= True)
    x = conv_block(x, neuron_num= 256, kernel_size= (3, 3), use_bias= True)

    # Define the converltional block 5
    x = conv_block(x, neuron_num= 512, kernel_size= (3, 3), use_bias= True, strides= (2, 2), with_conv_short_cut= True)
    x = conv_block(x, neuron_num= 512, kernel_size= (3, 3), use_bias= True)
    x = conv_block(x, neuron_num= 512, kernel_size= (3, 3), use_bias= True)
    x = K.layers.AveragePooling2D(pool_size=(7, 7))(x)
    return x


def model(dim_w,dim_h):
    First = K.layers.Input(shape=(dim_w,dim_h,3),name="input1")
    Second = K.layers.Input(shape=(dim_w,dim_h,3),name="input2")

    # x1 = K.layers.MaxPool2D(pool_size=(2,2),strides=2)(First)
    x1 = K.layers.Conv2D(128,kernel_size=(3,3), strides=2,padding='same')(First)
    x1 = K.layers.BatchNormalization()(x1)
    x1 = K.layers.LeakyReLU()(x1)
    # x1 = K.layers.Conv2D(256,kernel_size=(3,3), strides=2,padding='same')(x1)
    # x1 = K.layers.BatchNormalization()(x1)
    # x1 = K.layers.ReLU()(x1)
    x1 = ResNet34(x1,"x1")

    # x2 = K.layers.MaxPool2D(pool_size=(2,2),strides=2)(Second)
    x2 = K.layers.Conv2D(128,kernel_size=(3,3), strides=2,padding='same')(Second)
    x2 = K.layers.BatchNormalization()(x2)
    x2 = K.layers.LeakyReLU()(x2)
    # x2 = K.layers.Conv2D(256,kernel_size=(3,3), strides=2,padding='same')(x2)
    # x2 = K.layers.BatchNormalization()(x2)
    # x2 = K.layers.ReLU()(x2)
    x2 = ResNet34(x2,"x2")

    x = K.layers.concatenate([x1,x2])

    x = K.layers.Flatten()(x)
    x = K.layers.Dense(6,name='Output')(x)
    poseModel = K.Model([First,Second],x)

    return poseModel

def loss_fn(y_true,y_pre):
    loss_value_translation = K.backend.square(y_true[-1,0:3]-y_pre[-1,0:3])
    loss_value_rotation = 1/5.7*K.backend.square(y_true[-1,3:6]-y_pre[-1,3:6])
    loss_value = K.backend.mean(loss_value_translation + loss_value_rotation)

    # loss_value = K.backend.mean(K.backend.square(y_true-y_pre))
    # tf.print(y_pre)
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
            if epoch >= 30:
                lr = lr*self.alpha
        K.backend.set_value(self.model.optimizer.lr,K.backend.get_value(lr))
        if self.verbose > 0:
            print(f"Current learning rate is {lr}")
        #save the model
        if epoch % 20 == 0 and epoch != 0:
            self.model.save("model.h5")

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
        # self.optm = K.optimizers.Adam(1e-4)
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
            index = 0
            lr = float(self.optm.lr)
            tf.print(">>> [Epoch is %s/%s]"%(step,self.dataset.epochs))
            for (x1,x2),y in self.dataset.traindata:
                with tf.GradientTape() as tape:
                    prediction = self.poseModel([x1,x2])
                    # y = tf.cast(y,dtype=prediction.dtype)
                    loss = loss + loss_fn(y,prediction)
                gradients = tape.gradient(loss,self.poseModel.trainable_variables)
                self.optm.apply_gradients(zip(gradients,self.poseModel.trainable_variables))

                index = index + 1
                sys.stdout.write('--------train loss is %.5f-----'%(loss/float(index)))
                sys.stdout.write('\r')
                sys.stdout.flush()

            index_val = 0
            # 测试
            for (x1,x2),y in self.dataset.testdata:
                prediction = self.poseModel([x1,x2])
                val_loss = val_loss + loss_fn(y,prediction)
                index_val = index_val + 1
            tf.print("The loss is %s,the learning rate is : %s, test loss is %s]"%(np.array(loss/float(index)),lr,val_loss/float(index_val)))
            K.backend.set_value(self.optm.lr,K.backend.get_value(lr*0.99))
            if step%40==0:
                self.save_model()

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
    im1 = cv.imread("imagesNDI/0.jpg")
    im1 = cv.resize(im1,(512,512))
    im1 = np.array(im1,np.float).reshape((1,512,512,3))/255.0
    im2 = cv.imread("imagesNDI/1.jpg")
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
    
