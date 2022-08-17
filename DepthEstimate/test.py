import tensorflow as tf
import tensorflow.keras as K
import numpy as np
import cv2 as cv
import time
import sys
from dataset import datasets
from net import model

def test(dataset:datasets):
    #***************picture test************************
    im1 = cv.imread(np.array(dataset.test_pose_list)[0][0])
    im1 = cv.resize(im1,(512,512))
    im1 = np.array(im1,np.float).reshape((1,512,512,3))/255.0
    im2 = cv.imread(np.array(dataset.test_pose_list)[0][1])
    im2 = cv.resize(im2,(512,512))
    im2 = np.array(im2,np.float).reshape((1,512,512,3))/255.0
    im3 = cv.imread(np.array(dataset.test_pose_list)[0][2],cv.IMREAD_UNCHANGED)
    im3 = np.array(im3,np.uint16)
    # loading the depth by tf
    groudTruth = tf.io.read_file(np.array(dataset.test_pose_list)[0][2])
    groudTruth = tf.image.decode_png(groudTruth,channels=1,dtype=tf.uint16) #此处为jpeg格式
    groudTruth = np.array(groudTruth,np.uint16)
    groudTruth = np.reshape(groudTruth,newshape=(720,1280))

    print("Is the values of tf loading equal that of cv loading? ",(groudTruth == im3).all())
    #Network predict
    posemodel = K.models.load_model(dataset.model_path,compile=False)
    # posemodel = model(dataset.dim_w,dataset.dim_h)
    # posemodel.load_weights("model.h5")
    pose = posemodel([im1,im2])
    pose = pose * dataset.scale
    pose = np.reshape(pose,newshape=(512,512))
    pose = np.array(cv.resize(pose,(1280,720)),np.uint16)

    heatmapT = None
    heatmap = None
    heatmapTF = None
    heatmap = cv.normalize(pose, heatmap, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_8U)
    print("The max value in Network is :",np.max(pose))
    # load the groud trueth
    heatmapT = cv.normalize(im3, heatmapT, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_8U)
    print("The max value in groudT is :",np.max(im3))
    # load the image of the truth by tensorflow
    heatmapTF = cv.normalize(groudTruth, heatmapTF, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_8U)
    print("The max value in groudTF is :",np.max(groudTruth))

    cv.imshow("DepthNetwork",heatmap)
    cv.imshow("DepthGroundTruth",heatmapT)
    cv.imshow("DepthGroundTruthTF",heatmapTF)
    # cv.imwrite("test.jpg",pose)
    cv.waitKey(0)
    cv.destroyAllWindows()

    #**************video test*****************
    cap = cv.VideoCapture(6)
    cap.set(cv.CAP_PROP_FRAME_WIDTH,1920)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT,1080)

    depth = None
    depthDisplay = None

    def OnMouseAction(event,x,y,flags,param):
        if event == cv.EVENT_LBUTTONDOWN:
            print(depth[y,x])

    cv.namedWindow('heat')
    cv.setMouseCallback('heat',OnMouseAction)

    while True:
        success, frame = cap.read()
        frameL = frame[0:540,0:1920]
        frameR = frame[540:1080,0:1920]

        frameL = cv.resize(frameL,(512,512))
        frameL = np.array(frameL,np.float).reshape((1,512,512,3))/255.0

        frameR = cv.resize(frameR,(512,512))
        frameR = np.array(frameR,np.float).reshape((1,512,512,3))/255.0

        tic1 = time.perf_counter()
        depth = posemodel([frameL,frameR])
        toc1 = time.perf_counter()
        timeConsumption = toc1-tic1
        print(timeConsumption)
        
        depth = np.reshape(depth,newshape=(512,512))*dataset.scale
        depth = np.array(cv.resize(depth,(1280,720)),np.uint16)

        heatmaxp = None
        heatmap = cv.normalize(depth, heatmap, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_8U)
        depthDisplay = np.array(heatmap,np.uint8)
        heatMap = cv.applyColorMap(heatmap,cv.COLORMAP_JET)
        cv.imshow("heat",heatMap)
        cv.imshow("depthDispaly",depthDisplay)
        cv.imshow("depth",depth)
        if cv.waitKey(1) == 27:
            break