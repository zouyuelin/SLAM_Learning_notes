import tensorflow as tf
import tensorflow.keras as K
import numpy as np
import cv2 as cv
import time
import sys
from dataset import datasets
from net import model
from test import test


class learningDecay(K.callbacks.Callback):
    def __init__(self,dataset,schedule=None,alpha=1,verbose=0):
        super().__init__()
        self.dataset = dataset
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
        if epoch % 10 == 0 and epoch != 0:
            self.model.save("model.h5")
            self.model.save(self.dataset.model_path)

def loss_fn(y_true,y_pre):
    # smooth l1 loss
    discriminateV = K.backend.abs(y_true - y_pre)
    loss_value = tf.where(discriminateV < 1,0.5*K.backend.square(y_true - y_pre),K.backend.abs(y_true - y_pre)-0.5)
    loss_value = K.backend.mean(loss_value)
    return loss_value

def scheduler(epoch, lr):
    if epoch < 10:
        return lr
    else:
        return lr * tf.math.exp(-0.1) 

def errors(y_true,y_pre):
    error = K.backend.abs(y_true - y_pre)*25.60
    error = K.backend.mean(error)
    return error   

class Posenet:
    def __init__(self,dataset:datasets):
        self.dataset = dataset
        self.build()

    def build(self):
        self.poseModel = model(self.dataset.dim_w,self.dataset.dim_h)

        self.optm = K.optimizers.RMSprop(self.dataset.learning_rate,momentum=0.9) #,decay=1e-5/self.dataset.epochs
        # self.optm = K.optimizers.Adam(1e-4)
        self.decayCallback = learningDecay(self.dataset,schedule = None,alpha = 0.99,verbose = 1)
        decayCallbackScheduler = K.callbacks.LearningRateScheduler(scheduler)
        self.callbacks = [decayCallbackScheduler]

        try:
            print("************************loading the model weights***********************************")
            self.poseModel.load_weights("model.h5")
        except:
            pass

    def train_fit(self):
        self.poseModel.compile(optimizer=self.optm,loss=loss_fn,metrics=['accuracy',errors])
        # print the structure of the network
        self.poseModel.summary()

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



if __name__ == "__main__":
    dataset = datasets("images")
    posenet = Posenet(dataset)
    posenet.train_fit()
    # posenet.train_gradient() #利用 apply_gradient的方式训练
    posenet.save_model()
    test(dataset)
