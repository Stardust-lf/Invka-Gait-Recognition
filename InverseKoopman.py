import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
from Utils import debug_print_shape,getMaskForSize,gen_expand_matrix,show_sample_img
import random
import sys
import pickle

STD_LEN = 12
NUM_EPOCHS = 2000
NUM_DENSE_UNITS  = 2048
FIGURE_SEZE = 64
LOAD_WIGHTS = False
TRAIN_PRED_LAYER = False
BATCH_SIZE = 4
def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    random.seed(seed)
setup_seed(1024)
class MyModel(tf.keras.Model):
    def __init__(self,initializer):
        self.fine_tune=False
        self.batch_size=BATCH_SIZE
        super(MyModel, self).__init__()
        pre_mask = tf.convert_to_tensor(getMaskForSize(FIGURE_SEZE),dtype=tf.bool)
        pre_mask = tf.expand_dims(pre_mask,0)
        pre_mask = tf.expand_dims(pre_mask,-1)
        self.mask = pre_mask
        self.demask = tf.logical_not(pre_mask)
        pre_expandML,pre_expandMR = gen_expand_matrix(FIGURE_SEZE)
        self.expandML = tf.convert_to_tensor(pre_expandML,dtype=tf.float32)
        self.expandMR = tf.convert_to_tensor(pre_expandMR,dtype=tf.float32)
        self.kopMatrix = tf.Variable(initializer(shape=[FIGURE_SEZE,FIGURE_SEZE],dtype=tf.float32),name='Koopman Matrics',trainable=True)
        self.dense1 = tf.keras.layers.Dense(units=2048, name='dense1',trainable=not self.fine_tune)
        self.dense3 = tf.keras.layers.Dense(units=2048, name='dense3',trainable=not self.fine_tune)
        self.BatchNormalization1 = tf.keras.layers.BatchNormalization(name='Norm1',trainable=not self.fine_tune)
        self.BatchNormalization3 = tf.keras.layers.BatchNormalization(name='Norm3',trainable=not self.fine_tune)

    def computeF(self, u, cycleCount):
        s = tf.reshape(u, shape=[-1, FIGURE_SEZE, int(FIGURE_SEZE / 2),1])
        s = tf.reshape(s, [cycleCount * STD_LEN, -1])
        s = self.BatchNormalization1(s)
        s = self.dense1(s)
        s = tf.reshape(s, [cycleCount * STD_LEN, FIGURE_SEZE, int(FIGURE_SEZE / 2)])
        return s

    def computeG(self, u, cycleCount):
        t = tf.reshape(u, shape=[-1, FIGURE_SEZE, int(FIGURE_SEZE / 2),1])
        t = tf.reshape(t, [cycleCount * STD_LEN, -1])
        t = self.BatchNormalization3(t)
        t = self.dense3(t)
        t = tf.reshape(t, [cycleCount * STD_LEN, FIGURE_SEZE, int(FIGURE_SEZE/2)])
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
        cycleCount = inputs.shape[0]
        x = inputs
        x_next = inputs[:,1:]
        phiX = self.encoder(x,cycleCount)
        dephiPhiX = self.decoder(phiX,cycleCount)
        phiX_next = phiX[:,1:]

        proce_phiX = tf.reshape(phiX,shape=[self.batch_size,-1,STD_LEN,FIGURE_SEZE,FIGURE_SEZE])

        proce_phiX = tf.transpose(proce_phiX,[1,0,2,3,4])

        kPhix = tf.matmul(proce_phiX,self.kopMatrix)
        kPhix = tf.reshape(kPhix,[-1,STD_LEN,FIGURE_SEZE,FIGURE_SEZE])
        dePhiKPhiX = self.decoder(kPhix,cycleCount)
        return x,phiX,dephiPhiX,kPhix,dePhiKPhiX,x_next,phiX_next

def getDistance(vector1,vector2):
    return tf.sqrt(tf.reduce_sum(tf.square(vector2-vector1)))

def train_step(indexs,images,model,optimizer):
    with tf.GradientTape() as t:
        x, phiX, dephiPhiX, kPhix, dePhiKPhiX, x_next, phiX_next= model(images, indexs)
        loss0 = getDistance(phiX,dephiPhiX)
        loss1 = getDistance(kPhix[:,:-1],phiX_next)
        loss2 = getDistance(dePhiKPhiX[:,:-1],x_next)
        loss_avg = tf.reduce_mean(loss1 + loss2)
        variables = model.trainable_variables
        grads = t.gradient(loss_avg,variables)
        # print(variables[-1])
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

def computeCoder(tag,angle):
    data = load_variavle('data124_{0}_{1}'.format(tag,angle))
    data = np.array(data, dtype=object)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model = MyModel(tf.random_normal_initializer(mean=1., stddev=2.))
    data = preprocess(data)
    with open('bestLoss2.txt', 'r+') as f:
        if (LOAD_WIGHTS == False):
            f.seek(0)
            f.truncate()
            f.write(str(99999))
            bestLoss = 99999
        else:
            bestLoss = float(f.read())
        for i in range(NUM_EPOCHS):
            if (type(data[i % 10]) != np.ndarray):
                continue
            NUM_CLASSES = len(data)
            indexs = random.sample(range(NUM_CLASSES), BATCH_SIZE)
            inputData = data[indexs]
            lenList = []
            for sample in inputData:
                lenList.append(sample.shape[0])
            length = np.min(lenList)
            adata = []
            for sample in inputData:
                adata.append(sample[np.random.choice(len(sample), length, False)])
            loss0, loss1, loss2 = train_step(indexs, adata, model, optimizer)
            if i == 0:
                model.summary()
            lossSum = loss1 + loss2
            if (lossSum < bestLoss and lossSum != 0):
                print('Saving checkpoint', str(lossSum.numpy()), 'now epoch', str(i))
                f.seek(0)
                f.truncate()
                f.write(str(lossSum.numpy()))
                bestLoss = lossSum.numpy()
                model.save_weights('coder_fin_{0}_{1}'.format(tag,angle))
            if (i % 100 == 0):
                print('Epoch{3}          loss0:{0}  loss1:{1}  loss2:{2}'.format(loss0, loss1, loss2, i))


if __name__ == '__main__':
    data = load_variavle('data124_OULP')
    data = np.array(data,dtype=object)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model = MyModel(tf.random_normal_initializer(mean=1., stddev=2.))
    data = preprocess(data)
    loss0L = []
    loss1L = []
    loss2L = []
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
            indexs = random.sample(range(len(data)),BATCH_SIZE)
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
            lossSum = loss1 + loss2
            if (lossSum < bestLoss and lossSum!=0):
                print('Saving checkpoint', str(lossSum.numpy()), 'now epoch', str(i))
                f.seek(0)
                f.truncate()
                f.write(str(lossSum.numpy()))
                bestLoss = lossSum.numpy()
                model.save_weights('coder_fin_OULP')
            if(i%100 == 0):
                print('Epoch{3}          loss0:{0}  loss1:{1}  loss2:{2}'.format(loss0, loss1, loss2, i))




