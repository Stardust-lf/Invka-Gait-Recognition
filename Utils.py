import numpy as np
import tensorflow as tf
import os
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import pickle
import tensorflow_addons as tfa
def load_variavle(filename):
  f=open(filename,'rb')
  r=pickle.load(f)
  f.close()
  return r

def cut(image):
    image = np.array(image)
    height_min = (image.sum(axis=1) != 0).argmax()
    height_max = ((image.sum(axis=1) != 0).cumsum()).argmax()
    width_min = (image.sum(axis=0) != 0).argmax()
    width_max = ((image.sum(axis=0) != 0).cumsum()).argmax()
    head_top = image[height_min, :].argmax()
    size = height_max - height_min
    temp = np.zeros((size, size))
    l1 = head_top - width_min
    r1 = width_max - head_top
    flag = False
    if size <= width_max - width_min or size // 2 < r1 or size // 2 < l1:
        flag = True
        return temp, flag
    temp[:, (size // 2 - l1):(size // 2 + r1)] = image[height_min:height_max, width_min:width_max]
    return temp, flag

def load_preprosess_image(input_path):
    image = tf.io.read_file(input_path)
    image = tf.image.decode_jpeg(image,channels=3)
    image = tf.image.resize(image,[256,256])
    image = tf.cast(image,tf.float32)
    image = image/255
    return image

def getImgs(path):
    imgTensor = None
    for root,dirs,files in os.walk(path):
        for file in files:
            absPath = os.path.join(root,file)
            img = Image.open(absPath)
            img = np.array(img)
            cutImg = cv2.resize(img, (64, 44))
            img = tf.convert_to_tensor(cutImg,dtype=tf.float32)
            img = tf.expand_dims(img,0)
            img = tf.expand_dims(img,0)

            if imgTensor is None:
                imgTensor = img
            else:
                imgTensor = tf.concat([imgTensor,img],axis=1)
    return imgTensor

def cutRound(tensor):
    tensor = tensor[:,:40,:,:]
    return tensor
def load_one_class(root,className,angle):
    imgTensor = None

    for i in range(1,7):
        path = root + '\\nm-0{0}\\'.format(i) + angle + '\\'
        print(path)
        img = getImgs(path)
        if img == None :
            continue
        if img.shape[1]<40:
            continue
        if imgTensor is None:
            imgTensor = cutRound(img)
        else:
            imgTensor = tf.concat([imgTensor,cutRound(img)],axis=0)
    if imgTensor == None:
        labels = None
    else:
        labels = tf.constant(shape=[imgTensor.shape[0]],value=className,dtype=tf.int8)
    return imgTensor, labels

def load_all_classes(class_num,angle):
    data,labels = load_one_class('D:\\DataSets\\CASIA-B\\001',className=1,angle = angle)
    for i in range(2,class_num+1):
        num = ''
        if i < 10:
            num = '00' + str(i)
        elif i < 100:
            num = '0' + str(i)
        else:
            num = str(i)
        dataTemp,labelsTemp = load_one_class('D:\\DataSets\\CASIA-B\\{0}\\'.format(num,num),className=i,angle = angle)
        if labelsTemp != None:
            data = tf.concat([dataTemp,data],axis=0)
            labels = tf.concat([labelsTemp, labels], axis=0)

    return data,labels

def debug_print_shape(val,tag,debug):
    if debug==True:
        print(val.shape,tag)

def get6464mask():
    trList = [True] * 64
    flList = [False] * 64
    fin = []
    for i in range(64):
        if i % 2 == 0:
            fin.append(trList)
        else:
            fin.append(flList)
    fin = np.array(fin)
    fin = fin.T
    for j in range(32):
        fin[j * 2] = np.roll(fin[j * 2], 1)
    return fin
def getMaskForSize(size):
    trList = [True] * size
    flList = [False] * size
    fin = []
    for i in range(size):
        if i % 2 == 0:
            fin.append(trList)
        else:
            fin.append(flList)
    fin = np.array(fin)
    fin = fin.T
    for j in range(int(size/2)):
        fin[j * 2] = np.roll(fin[j * 2], 1)
    return fin

def gen_expand_matrix(figSize):
    x = np.zeros(shape=[int(figSize/2),figSize])
    for i in range(len(x)):
        x[i][2*i+1] = 1
    y = np.zeros(shape=[int(figSize/2),figSize])
    for j in range(len(y)):
        y[j][2*j] = 1
    return x,y

def show_sample_img(tensor):
    plt.imshow(tensor[0].numpy())
    plt.show()