import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
from Utils import debug_print_shape,get6464mask,gen_expand_matrix,show_sample_img,getMaskForSize
import random
import sys
import pickle
from tensorflow import optimizers
STD_LEN = 12
NUM_EPOCHS = 400
NUM_DENSE_UNITS  = 2048
FIGURE_SEZE = 64
NUM_CLASSES = 124
LOAD_WIGHTS = False
TRAIN_PRED_LAYER = False
BATCH_SIZE = 1
# LENLIST = [20, 23, 25, 27, 4, 18, 22, 26, 23, 22, 22, 29, 29, 23, 18, 22, 21, 17, 18, 18, 23, 21, 21, 27, 21, 23, 23, 25, 22, 19, 21, 23, 20, 27, 21, 26, 19, 23, 20, 26, 23, 18, 24, 24, 26, 19, 28, 25, 18, 18, 19, 21, 23, 22, 23, 20, 23, 21, 23, 28, 19, 21, 18, 18, 23, 19, 23, 23, 21, 19, 25, 20, 21, 19, 22, 24, 23, 21, 20, 25, 19, 29, 23, 21, 19, 21, 22, 21, 18, 21, 17, 28, 23, 22, 20, 21, 22, 18, 20, 23, 24, 28, 20, 19, 16, 22, 23, 23, 25, 30, 22, 22, 21, 19, 23, 20, 22, 30, 25, 23, 21, 18, 19, 20]
# LENLIST = [21, 23, 25, 26, 18, 22, 27, 22, 23, 22, 28, 27, 22, 18, 24, 20, 17, 20, 18, 24, 21, 20, 27, 20, 23, 22, 25, 20, 19, 23, 23, 21, 27, 21, 25, 19, 24, 22, 26, 22, 18, 25, 27, 24, 19, 27, 25, 17, 17, 20, 20, 23, 22, 22, 18, 22, 20, 22, 29, 17, 21, 19, 17, 25, 20, 23, 23, 20, 20, 24, 21, 20, 18, 23, 24, 23, 21, 18, 24, 19, 29, 23, 22, 19, 21, 22, 22, 18, 22, 17, 27, 26, 22, 20, 21, 19, 19, 21, 23, 23, 28, 18, 20, 15, 22, 22, 22, 25, 30, 22, 22, 20, 21, 21, 23, 22, 28, 25, 23, 21, 21, 18, 20]
#LENLIST = [7, 7, 10, 9, 3, 5, 8, 9, 7, 8, 7, 9, 8, 8, 7, 8, 6, 6, 6, 6, 8, 7, 7, 10, 7, 8, 8, 8, 8, 7, 8, 8, 7, 8, 5, 8, 6, 8, 7, 8, 7, 6, 8, 8, 8, 6, 8, 8, 8, 6, 7, 7, 6, 7, 8, 7, 8, 6, 6, 9, 7, 7, 6, 6, 8, 7, 8, 6, 7, 8, 8, 7, 6, 5, 7, 6, 7, 7, 7, 8, 6, 10, 8, 7, 6, 8, 7, 8, 6, 8, 6, 10, 8, 6, 6, 7, 8, 5, 6, 8, 7, 9, 5, 6, 6, 8, 6, 7, 8, 10, 8, 7, 7, 6, 8, 8, 6, 10, 8, 7, 8, 7, 6, 5]
#LENLIST = [24, 24, 25, 26, 12, 18, 21, 28, 24, 22, 24, 31, 27, 23, 24, 25, 23, 17, 22, 18, 24, 23, 23, 28, 23, 26, 25, 24, 23, 22, 24, 27, 24, 28, 24, 28, 22, 25, 25, 24, 22, 18, 27, 26, 26, 23, 33, 29, 22, 18, 17, 23, 19, 24, 25, 21, 25, 22, 24, 27, 22, 23, 22, 20, 25, 18, 25, 25, 21, 22, 27, 26, 24, 20, 23, 23, 26, 23, 24, 26, 21, 33, 24, 24, 21, 24, 24, 22, 19, 23, 21, 30, 27, 24, 21, 25, 22, 21, 22, 25, 24, 26, 18, 19, 18, 22, 27, 26, 28, 27, 23, 26, 23, 18, 25, 22, 22, 28, 27, 27, 23, 23, 21, 23]
# above is for dataset72
# LENLIST = [28, 42, 32, 39, 14, 30, 26, 36, 29, 31, 29, 38, 32, 30, 31, 37, 28, 20, 37, 27, 34, 37, 31, 58, 32, 38, 36, 46, 42, 28, 32, 27, 33, 43, 35, 38, 40, 29, 34, 34, 29, 22, 33, 29, 32, 23, 42, 32, 30, 25, 33, 25, 27, 28, 31, 27, 26, 26, 24, 23, 27, 29, 24, 26, 39, 29, 33, 9, 25, 25, 31, 32, 31, 36, 27, 20, 35, 25, 17, 26, 27, 48, 31, 18, 23, 23, 29, 12, 29, 23, 27, 39, 30, 32, 24, 37, 20, 24, 23, 25, 23, 34, 24, 21, 25, 38, 33, 38, 27, 31, 23, 35, 34, 22, 34, 39, 28, 37, 33, 23, 33, 23, 27, 26]
# above is for dataset36
# LENLIST = [25, 36, 25, 31, 0, 25, 25, 28, 27, 28, 30, 30, 37, 36, 28, 27, 32, 16, 25, 27, 28, 34, 27, 44, 33, 29, 28, 34, 34, 36, 31, 24, 29, 36, 29, 29, 37, 28, 28, 39, 23, 13, 33, 28, 32, 26, 39, 33, 27, 24, 24, 22, 24, 29, 28, 25, 28, 21, 22, 27, 27, 25, 19, 22, 34, 28, 31, 18, 24, 23, 31, 27, 30, 29, 22, 22, 31, 21, 19, 21, 24, 40, 29, 19, 21, 25, 27, 31, 22, 24, 19, 34, 33, 27, 35, 21, 22, 23, 23, 21, 26, 21, 20, 18, 35, 24, 28, 37, 31, 23, 29, 27, 21, 29, 33, 28, 30, 30, 20, 33, 22, 29, 22]
# above is for dataset00
#LENLIST = [34, 42, 37, 37, 24, 31, 31, 43, 28, 31, 33, 39, 42, 37, 33, 36, 28, 22, 33, 34, 36, 34, 32, 51, 32, 41, 28, 35, 42, 34, 31, 29, 33, 40, 37, 37, 42, 36, 33, 38, 28, 24, 34, 31, 40, 28, 38, 36, 33, 26, 32, 28, 23, 31, 32, 29, 26, 22, 26, 28, 23, 27, 28, 25, 38, 27, 34, 19, 22, 24, 31, 27, 29, 36, 28, 21, 45, 23, 20, 26, 29, 45, 34, 23, 28, 24, 26, 12, 29, 28, 21, 37, 33, 28, 31, 23, 26, 24, 29, 21, 31, 19, 22, 16, 36, 33, 35, 29, 31, 29, 36, 33, 20, 37, 33, 30, 33, 29, 24, 39, 28, 28, 26]
# above is for dataset18
# LENLIST = [7, 7, 9, 9, 0, 5, 8, 10, 6, 8, 9, 9, 7, 9, 8, 8, 8, 6, 8, 6, 8, 8, 8, 10, 8, 9, 8, 7, 8, 7, 8, 7, 9, 9, 7, 9, 8, 8, 8, 10, 7, 6, 10, 9, 10, 7, 10, 8, 8, 4, 7, 7, 8, 7, 8, 7, 9, 6, 7, 10, 7, 8, 6, 6, 8, 8, 8, 8, 7, 8, 9, 7, 8, 6, 7, 6, 9, 8, 8, 8, 7, 12, 8, 8, 7, 9, 7, 7, 6, 7, 6, 10, 8, 7, 7, 8, 6, 5, 8, 7, 8, 7, 7, 6, 8, 8, 8, 8, 11, 8, 8, 7, 7, 8, 8, 8, 10, 10, 7, 7, 7, 7, 7]
# above is for datasetbg72
#LENLIST = [11, 11, 13, 14, 3, 7, 9, 16, 7, 8, 13, 9, 11, 12, 10, 10, 8, 4, 6, 11, 11, 12, 11, 18, 11, 12, 11, 16, 13, 11, 10, 11, 9, 10, 7, 14, 6, 12, 10, 8, 7, 7, 9, 11, 15, 6, 11, 8, 9, 7, 8, 8, 11, 9, 9, 7, 10, 8, 7, 6, 6, 10, 7, 7, 7, 11, 8, 13, 10, 7, 13, 11, 7, 9, 10, 8, 13, 8, 4, 10, 10, 17, 6, 10, 9, 10, 9, 9, 9, 6, 10, 12, 10, 10, 6, 9, 7, 7, 7, 9, 9, 9, 7, 6, 12, 11, 11, 7, 11, 9, 13, 10, 7, 11, 7, 6, 13, 6, 8, 13, 7, 9, 7]
# above is for datasetbg36
# LENLIST = [9, 7, 11, 11, 0, 8, 10, 11, 9, 9, 9, 10, 11, 12, 8, 8, 10, 4, 8, 8, 11, 10, 9, 14, 12, 10, 7, 12, 14, 13, 8, 8, 10, 11, 8, 13, 8, 7, 8, 10, 8, 11, 9, 11, 13, 7, 12, 7, 9, 6, 8, 8, 7, 7, 9, 8, 10, 6, 7, 8, 9, 8, 7, 6, 7, 7, 10, 9, 8, 5, 12, 9, 7, 8, 7, 8, 9, 11, 9, 5, 8, 14, 6, 8, 7, 7, 8, 10, 7, 9, 7, 9, 8, 8, 8, 8, 7, 6, 7, 8, 11, 7, 8, 6, 11, 8, 10, 11, 11, 9, 8, 7, 6, 11, 9, 11, 10, 8, 8, 11, 7, 9, 7]
#above is for datasetbg00
# LENLIST = [8, 13, 11, 12, 0, 6, 10, 12, 9, 10, 10, 8, 14, 11, 10, 8, 11, 5, 6, 9, 8, 9, 11, 19, 11, 11, 11, 13, 9, 10, 8, 8, 9, 15, 9, 11, 9, 9, 8, 9, 7, 5, 9, 11, 11, 7, 10, 11, 7, 7, 7, 8, 6, 8, 8, 8, 9, 7, 10, 9, 8, 10, 8, 6, 10, 9, 7, 7, 7, 8, 11, 9, 8, 10, 10, 6, 10, 7, 5, 8, 6, 10, 7, 9, 9, 10, 7, 8, 8, 8, 6, 13, 9, 9, 10, 10, 6, 8, 7, 9, 9, 7, 7, 7, 10, 10, 11, 7, 10, 11, 8, 10, 8, 10, 10, 9, 9, 6, 7, 8, 10, 7, 6]
# above is for datasetcl00
#LENLIST = [9, 8, 12, 15, 5, 5, 13, 11, 8, 8, 12, 12, 10, 11, 9, 9, 9, 9, 11, 8, 11, 10, 11, 18, 13, 16, 11, 14, 10, 9, 10, 11, 12, 16, 9, 12, 9, 10, 12, 8, 7, 5, 11, 12, 10, 7, 9, 0, 5, 6, 7, 8, 9, 7, 9, 10, 8, 6, 9, 6, 8, 10, 10, 8, 8, 9, 9, 8, 8, 6, 10, 11, 7, 11, 9, 8, 15, 6, 5, 7, 5, 16, 8, 12, 7, 10, 9, 6, 8, 6, 7, 14, 9, 7, 10, 9, 6, 6, 9, 7, 12, 6, 8, 8, 9, 12, 12, 9, 12, 8, 11, 13, 8, 9, 11, 10, 10, 7, 8, 9, 8, 11, 7]
# above is for datasetcl36
LENLIST = [8, 7, 8, 9, 5, 6, 8, 10, 8, 8, 7, 10, 6, 8, 8, 7, 8, 7, 7, 6, 8, 7, 8, 8, 7, 7, 9, 7, 8, 7, 9, 8, 9, 9, 8, 9, 7, 7, 8, 10, 6, 5, 9, 9, 9, 7, 9, 9, 6, 6, 7, 8, 6, 7, 7, 7, 6, 6, 8, 9, 8, 8, 7, 7, 9, 7, 8, 7, 8, 8, 10, 8, 7, 6, 8, 7, 8, 6, 7, 8, 7, 10, 7, 8, 7, 7, 7, 6, 8, 6, 6, 12, 8, 7, 5, 7, 6, 6, 7, 8, 8, 6, 7, 6, 8, 8, 8, 7, 11, 6, 9, 7, 6, 8, 7, 8, 7, 10, 9, 8, 6, 6, 5]
# above is for datasetcl72
def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    random.seed(seed)
setup_seed(1024)
class MyModel(tf.keras.Model):
    def __init__(self,initializer):
        self.fine_tune=True

        self.batch_size=BATCH_SIZE
        super(MyModel, self).__init__()
        #[cycleCount * STD_LEN, FIGURE_SEZE, FIGURE_SEZE, 1]
        pre_mask = tf.convert_to_tensor(getMaskForSize(FIGURE_SEZE),dtype=tf.bool)
        pre_mask = tf.expand_dims(pre_mask,0)
        pre_mask = tf.expand_dims(pre_mask,-1)
        self.mask = pre_mask
        self.demask = tf.logical_not(pre_mask)
        pre_expandML,pre_expandMR = gen_expand_matrix(FIGURE_SEZE)
        self.expandML = tf.convert_to_tensor(pre_expandML,dtype=tf.float32)
        self.expandMR = tf.convert_to_tensor(pre_expandMR,dtype=tf.float32)
        self.kopMatrix = tf.Variable(initializer(shape=[FIGURE_SEZE,FIGURE_SEZE],dtype=tf.float32),name='Koopman Matrics',trainable=True)
        self.dense1 = tf.keras.layers.Dense(units=2048, name='dense1', trainable=not self.fine_tune)
        self.dense3 = tf.keras.layers.Dense(units=2048, name='dense3', trainable=not self.fine_tune)
        self.BatchNormalization1 = tf.keras.layers.BatchNormalization(name='Norm1', trainable=not self.fine_tune)
        self.BatchNormalization3 = tf.keras.layers.BatchNormalization(name='Norm3', trainable=not self.fine_tune)


    def computeF(self, u, cycleCount):
        s = tf.reshape(u, shape=[-1, FIGURE_SEZE, int(FIGURE_SEZE / 2), 1])
        s = tf.reshape(s, [cycleCount * STD_LEN, -1])
        s = self.BatchNormalization1(s)
        s = self.dense1(s)
        s = tf.reshape(s, [cycleCount * STD_LEN, FIGURE_SEZE, int(FIGURE_SEZE / 2)])
        return s

    def computeG(self, u, cycleCount):
        t = tf.reshape(u, shape=[-1, FIGURE_SEZE, int(FIGURE_SEZE / 2), 1])
        t = tf.reshape(t, [cycleCount * STD_LEN, -1])
        t = self.BatchNormalization3(t)
        t = self.dense3(t)
        t = tf.reshape(t, [cycleCount * STD_LEN, FIGURE_SEZE, int(FIGURE_SEZE / 2)])
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

def Create_folder(filename):
    filename = filename.strip()
    filename = filename.rstrip("\\")
    isExists = os.path.exists(filename)

    if not isExists:
        os.makedirs(filename)
        print(filename+"创建成功")
        return  True
    else:
        print(filename+"已存在")
        return False

def comp_mat(tag,angle):
    data = load_variavle('data124_{0}_{1}'.format(tag,angle))
    Create_folder('Folder_{0}_{1}'.format(tag,angle))
    data = np.array(data, dtype=object)
    data = preprocess(data)
    for sample in data:
        print(len(sample))

    optimizer = optimizers.Adam(learning_rate=0.01, epsilon=1e-4)
    model = MyModel(tf.random_normal_initializer(mean=1., stddev=2.))
    model.fine_tune = True
    model.batch_size = 1
    model([data[0]], [0])
    print('loading checkpoint')
    model.load_weights('coder_fin_{0}_{1}'.format(tag,angle))
    model.summary()
    print(len(data))
    for j in range(len(data)):
        sample = data[j]
        for i in range(len(sample)):
            epoch = 0
            while (epoch <= NUM_EPOCHS):
                epoch += 1
                loss0, loss1, loss2 = train_step([0], [tf.expand_dims(sample[i], axis=0)], model, optimizer)
                if epoch % 100 == 0:
                    print('Training epoch{3} for sample {4} cycle {5}(of{6}) loss0:{0}  loss1:{1}  loss2:{2}'.format(
                        loss0, loss1, loss2, epoch, j, i, len(sample)))
            print('Sample {} has been trained'.format(i))
            weigts_parm = model.trainable_variables
            kop_matrix = weigts_parm[0].value()
            kop_matrix = kop_matrix.numpy()
            print(kop_matrix.shape)
            with open('Folder_{0}_{1}/sample{2}_cycle{3}'.format(tag,angle,j, i), 'wb+') as f:
                pickle.dump(kop_matrix, f)
                plt.imshow(kop_matrix)
                plt.show()
            print('Loading ckpt')
            model.load_weights('coder_fin_{0}_{1}'.format(tag,angle))

if __name__ == '__main__':
    data = load_variavle('data124_cl_72')
    data = np.array(data,dtype=object)
    data = preprocess(data)
    for sample in data:
        print(len(sample))

    optimizer = optimizers.Adam(learning_rate=0.01, epsilon=1e-4)
    model = MyModel(tf.random_normal_initializer(mean=1., stddev=2.))
    model.fine_tune = True
    model.batch_size = 1
    model([data[0]],[0])
    print('loading checkpoint')
    model.load_weights('coder_fin_cl_72')
    model.summary()
    print(len(data))
    for j in range(124):
        sample = data[j]
        for i in range(LENLIST[j]):
            loss1 = 5000
            loss2 = 5000
            epoch = 0
            while(epoch<=800):
                epoch += 1
                loss0, loss1, loss2 = train_step([0], [tf.expand_dims(sample[i],axis=0)], model, optimizer)
                if epoch%100 == 0:
                    print('Training epoch{3} for sample {4} cycle {5}(of{6}) loss0:{0}  loss1:{1}  loss2:{2}'.format(loss0, loss1, loss2, epoch, j, i, len(sample)))
            print('Sample {} has been trained'.format(i))
            weigts_parm = model.trainable_variables
            kop_matrix = weigts_parm[0].value()
            kop_matrix = kop_matrix.numpy()
            print(kop_matrix.shape)
            with open('matrix64cl_72/sample{0}_cycle{1}'.format(j,i),'wb+') as f:
                pickle.dump(kop_matrix,f)
            print('Loading ckpt')
            model.load_weights('coder_fin_cl_72')






