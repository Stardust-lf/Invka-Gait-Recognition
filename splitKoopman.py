import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
from Utils import debug_print_shape,get6464mask,gen_expand_matrix,show_sample_img
import random
import sys
import pickle

STD_LEN = 12
KOOPMAN_SIZE = 2048
NUM_EPOCHS = 5000
NUM_DENSE_UNITS  = 2048
FIGURE_SEZE = 64
NUM_CLASSES = 123
NOW_INDEX = 0
LOAD_WIGHTS = False
BATCH_SIZE = 3
TRAIN_PRED_LAYER = False

class MyModel(tf.keras.Model):
    def __init__(self,initializer):
        super(MyModel, self).__init__()
        #[cycleCount * STD_LEN, FIGURE_SEZE, FIGURE_SEZE, 1]
        pre_mask = tf.convert_to_tensor(get6464mask(),dtype=tf.bool)
        pre_mask = tf.expand_dims(pre_mask,0)
        pre_mask = tf.expand_dims(pre_mask,-1)
        self.mask = pre_mask
        self.demask = tf.logical_not(pre_mask)
        pre_expandML,pre_expandMR = gen_expand_matrix(FIGURE_SEZE)
        self.expandML = tf.convert_to_tensor(pre_expandML,dtype=tf.float32)
        self.expandMR = tf.convert_to_tensor(pre_expandMR,dtype=tf.float32)

        # self.conv1 = tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=(1, 1), padding='SAME',
        #                                     data_format='channels_last', name='conv1')
        # self.conv2 = tf.keras.layers.Conv2D(filters=64, kernel_size=1, strides=(1, 1), padding='SAME',
        #                                     data_format='channels_last', name='conv2')
        # self.pool1 = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2, padding='VALID',
        #                                     data_format='channels_last', name='pool1')
        # self.conv4 = tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=(1, 1), padding='SAME',
        #                                     data_format='channels_last', name='conv4')
        # self.conv5 = tf.keras.layers.Conv2D(filters=64, kernel_size=1, strides=(1, 1), padding='SAME',
        #                                     data_format='channels_last', name='conv5')
        # self.pool2 = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2, padding='VALID',
        #                                        data_format='channels_last', name='pool2')
        self.dense1 = tf.keras.layers.Dense(units=4096, name='dense1')
        self.dense2 = tf.keras.layers.Dense(units=NUM_DENSE_UNITS, name='dense2')
        self.dense3 = tf.keras.layers.Dense(units=4096, name='dense3')
        self.dense4 = tf.keras.layers.Dense(units=NUM_DENSE_UNITS, name='dense4')

        self.denseKop = tf.keras.layers.Dense(units=4096, name='Kop dense')

        # self.dense5 = tf.keras.layers.Dense(units=NUM_DENSE_UNITS, name='dense5')
        # self.dense6 = tf.keras.layers.Dense(units=NUM_DENSE_UNITS, name='dense6')
        self.BatchNormalization1 = tf.keras.layers.BatchNormalization(name='Norm1')
        self.BatchNormalization2 = tf.keras.layers.BatchNormalization(name='Norm2')
        self.BatchNormalization3 = tf.keras.layers.BatchNormalization(name='Norm3')
        self.BatchNormalization4 = tf.keras.layers.BatchNormalization(name='Norm4')
        # self.dense5 = tf.keras.layers.Dense(units=1024, name='dense5',trainable=TRAIN_PRED_LAYER)
        # self.dense6 = tf.keras.layers.Dense(units=512, name='dense6',trainable=TRAIN_PRED_LAYER)
        # self.dense7 = tf.keras.layers.Dense(units=256, name='dense7',trainable=TRAIN_PRED_LAYER)
        # self.dense8 = tf.keras.layers.Dense(units=NUM_CLASSES, name='dense8',trainable=TRAIN_PRED_LAYER)
        # self.BatchNormalization5 = tf.keras.layers.BatchNormalization(name='Norm5',trainable=TRAIN_PRED_LAYER)
        # self.flatten = tf.keras.layers.Flatten(name='flatten',trainable=TRAIN_PRED_LAYER)

    def computeF(self, u, cycleCount):
        s = tf.reshape(u, shape=[-1, FIGURE_SEZE, int(FIGURE_SEZE / 2),1])
        # s = self.conv1(s)
        # s = self.conv2(s)
        #s = tf.reduce_mean(s,axis=-1)
        s = tf.reshape(s, [cycleCount * STD_LEN, -1])
        s = self.BatchNormalization1(s)
        s = self.dense1(s)
        s = self.BatchNormalization2(s)
        s = self.dense2(s)
        # s = self.dense3(s)
        # s = self.Normalization1(s)
        s = tf.reshape(s, [cycleCount * STD_LEN, FIGURE_SEZE, int(FIGURE_SEZE / 2)])
        return s

    def computeG(self, u, cycleCount):
        t = tf.reshape(u, shape=[-1, FIGURE_SEZE, int(FIGURE_SEZE / 2),1])
        # t = self.conv4(t)
        # t = self.conv5(t)
        #t = tf.reduce_mean(t,axis=-1)
        t = tf.reshape(t, [cycleCount * STD_LEN, -1])
        t = self.BatchNormalization3(t)
        t = self.dense3(t)
        t = self.BatchNormalization4(t)
        t = self.dense4(t)
        t = tf.reshape(t, [cycleCount * STD_LEN, FIGURE_SEZE, int(FIGURE_SEZE/2)])
        # t = self.dense6(t)
        return t

    def move_even(self,mat,direction):
        mat = tf.transpose(mat, [0, 2, 1])
        diag = tf.expand_dims(tf.linalg.diag([1.0, 0.0] * int(FIGURE_SEZE / 2)), 0)
        tileddiag = tf.tile(diag, [mat.shape[0], 1, 1])
        mL = tf.matmul(mat, tileddiag)
        diag_ = tf.expand_dims(tf.linalg.diag([0.0, 1.0] * int(FIGURE_SEZE / 2)), 0)
        tileddiag_ = tf.tile(diag_, [mat.shape[0], 1, 1])
        mR = tf.matmul(mat, tileddiag_)
        mT = tf.transpose(mL,[0,2,1])
        mB = tf.transpose(mR,[0,2,1])
        mB = tf.roll(mB, direction, axis=2)

        return mT + mB

    def mix(self,w1,w2):
        w1 = tf.matmul(w1, self.expandML)
        w2 = tf.matmul(w2, self.expandMR)
        w1 = self.move_even(w1, -1)
        w2 = self.move_even(w2, 1)
        result = w1 + w2
        return result

    def encoder(self,x,cycleCount):
        data = tf.reshape(x,[cycleCount*STD_LEN,FIGURE_SEZE,FIGURE_SEZE,1])
        masks = tf.tile(self.mask,[cycleCount*STD_LEN,1,1,1])
        demasks = tf.tile(self.demask,[cycleCount*STD_LEN,1,1,1])
        u1 = data[masks]
        u2 = data[demasks]
        u1 = tf.reshape(u1,shape=[cycleCount*STD_LEN,FIGURE_SEZE,int(FIGURE_SEZE/2)])
        u2 = tf.reshape(u2, shape=[cycleCount*STD_LEN, FIGURE_SEZE, int(FIGURE_SEZE/2)])
        v1 = u1 + self.computeF(u2, cycleCount)
        v2 = u2
        w1 = v1
        w2 = v2 + self.computeG(v1, cycleCount)
        result = self.mix(w1,w2)
        result = tf.reshape(result, [cycleCount, STD_LEN, FIGURE_SEZE, FIGURE_SEZE])
        return result

    def decoder(self,y,cycleCount):
        data = tf.reshape(y, [-1, FIGURE_SEZE, FIGURE_SEZE, 1])
        masks = tf.tile(self.mask, [cycleCount * STD_LEN, 1, 1, 1])
        demasks = tf.tile(self.demask, [cycleCount * STD_LEN, 1, 1, 1])
        w1 = data[masks]
        w2 = data[demasks]
        w1 = tf.reshape(w1,shape=[cycleCount*STD_LEN,FIGURE_SEZE,int(FIGURE_SEZE/2)])
        w2 = tf.reshape(w2, shape=[cycleCount*STD_LEN, FIGURE_SEZE, int(FIGURE_SEZE/2)])
        v1 = w1
        v2 = w2 - self.computeG(v1, cycleCount)
        u2 = v2
        u1 = v1 - self.computeF(u2, cycleCount)
        result = self.mix(u1,u2)
        result = tf.reshape(result,[cycleCount,STD_LEN,FIGURE_SEZE,FIGURE_SEZE])
        return result

    def call(self, inputs,indexs):

        inputs = tf.convert_to_tensor(inputs,dtype=tf.float32)

        inputs = tf.reshape(inputs,[-1,STD_LEN,FIGURE_SEZE,FIGURE_SEZE])
        #inputs = tf.expand_dims(inputs,axis=4)
        cycleCount = inputs.shape[0]
        x = inputs
        x_next = inputs[:,1:]
        phiX = self.encoder(x,cycleCount)
        dephiPhiX = self.decoder(phiX,cycleCount)
        phiX_next = phiX[:,1:]


        proc_phiX = tf.reshape(phiX,[-1,STD_LEN,FIGURE_SEZE*FIGURE_SEZE])
        kPhix = self.denseKop(proc_phiX)


        kPhix = tf.reshape(kPhix,[-1,STD_LEN,FIGURE_SEZE,FIGURE_SEZE])

        dePhiKPhiX = self.decoder(kPhix,cycleCount)

        return x,phiX,dephiPhiX,kPhix,dePhiKPhiX,x_next,phiX_next

def getDistance(vector1,vector2):
    return tf.sqrt(tf.reduce_sum(tf.square(vector2-vector1)))

def train_step(indexs,images,model,optimizer):
    with tf.GradientTape() as t:
        #temp = model.koopmanMatrics
        x, phiX, dephiPhiX, kPhix, dePhiKPhiX, x_next, phiX_next= model(images, indexs)
        loss0 = getDistance(phiX,dephiPhiX)
        loss1 = getDistance(kPhix[:,:-1],phiX_next)
        loss2 = getDistance(dePhiKPhiX[:,:-1],x_next)
        loss_avg = tf.reduce_mean(loss1 + loss2)
        variables = model.trainable_variables
        grads = t.gradient(loss_avg,variables)

        optimizer.apply_gradients(zip(grads,variables))

        return loss0, loss1, loss2

def preprocess(data):
    for i in range(len(data)):
        if(type(data[i]) == np.ndarray):
            data[i] /= 255.0
            data[i] = np.array(data[i],dtype=np.float32)
    return data

def load_variavle(filename):
  f=open(filename,'rb')
  r=pickle.load(f)
  f.close()
  return r

if __name__ == '__main__':
    setDir = 'data/pure'
    datasetIndex = random.randint(0,2)

    #data = load_variavle(os.path.join(setDir,'train{0}'.format(datasetIndex)))
    data = load_variavle('train0')
    #print(data)
    data = np.array(data,dtype=object)
    data = np.delete(data,4)
    print(type(data))
    # for i in range(len(data)):
    #     temp = data[i]
    #     data[i] = temp[:,[0,5,10,3,8,1,6,11,4,9,2,7]]
    optimizer = tf.keras.optimizers.Adagrad(learning_rate=0.001,epsilon=1e-4)
    #optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
    model = MyModel(tf.random_normal_initializer(mean=1., stddev=2.))
    data = preprocess(data)
    with open('bestLoss2.txt','r+') as f:
        if(LOAD_WIGHTS==False):
            f.seek(0)
            f.truncate()
            f.write(str(99999))
            bestLoss=99999
        else:
            bestLoss = float(f.read())
        for i in range(NUM_EPOCHS):
            if(type(data[i%10]) != np.ndarray):
                continue
            indexs = random.sample(range(NUM_CLASSES),BATCH_SIZE)
            inputData = data[indexs]
            lenList = []
            for sample in inputData:
                lenList.append(sample.shape[0])
            length = np.min(lenList)
            adata = []
            for sample in inputData:
                adata.append(sample[np.random.choice(len(sample),length,False)])
            loss0,loss1,loss2 = train_step(indexs,adata,model,optimizer)
            if i==0:
                model.summary()
                if(LOAD_WIGHTS):
                    print('loading checkpoint')
                    model.load_weights(os.path.join(setDir,'checkpoint{0}.h5'.format(datasetIndex)))
            lossSum = loss1 + loss2
            if (lossSum < bestLoss):
                print('Saving checkpoint', str(lossSum.numpy()), 'now epoch', str(i))
                f.seek(0)
                f.truncate()
                f.write(str(lossSum.numpy()))
                bestLoss = lossSum.numpy()
                model.save_weights('ckpt_encoder.h5'.format(datasetIndex))
            if(i%1 == 0):
                print('Epoch{3}          loss0:{0}  loss1:{1}  loss2:{2}'.format(loss0, loss1, loss2, i))

            NOW_INDEX += 1


