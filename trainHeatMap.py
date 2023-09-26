import pickle
from tensorflow.keras.layers import Dense, Flatten, BatchNormalization, MaxPooling2D, Dropout, Conv3D, MaxPooling3D, Conv2D

def load_variavle(filename):
  f=open(filename,'rb')
  r=pickle.load(f)
  f.close()
  return r

import pickle
import numpy as np
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Dense, Flatten, BatchNormalization, MaxPooling2D, Dropout, Conv3D, MaxPooling3D
LENLIST = [21, 23, 25, 26, 18, 22, 27, 22, 23, 22, 28, 27, 22, 18, 24, 20, 17, 20, 18, 24, 21, 20, 27, 20, 23, 22, 25, 20, 19, 23, 23, 21, 27, 21, 25, 19, 24, 22, 26, 22, 18, 25, 27, 24, 19, 27, 25, 17, 17, 20, 20, 23, 22, 22, 18, 22, 20, 22, 29, 17, 21, 19, 17, 25, 20, 23, 23, 20, 20, 24, 21, 20, 18, 23, 24, 23, 21, 18, 24, 19, 29, 23, 22, 19, 21, 22, 22, 18, 22, 17, 27, 26, 22, 20, 21, 19, 19, 21, 23, 23, 28, 18, 20, 15, 22, 22, 22, 25, 30, 22, 22, 20, 21, 21, 23, 22, 28, 25, 23, 21, 21, 18, 20]

y = []
for i in range(len(LENLIST)):
  y = y + [i]*LENLIST[i]
y = np.array(y)
dataList = []
data = load_variavle('heatMap')
print(data.shape,'heatMap')

inp = Input(data.shape)
x = Conv2D(filters=16,kernel_size=[3,3],strides=1)(inp)
x = Conv2D(filters=8,kernel_size=[3,3],strides=1)(x)
x = Flatten()(x)
x = Dense(2048)(x)
x = Dense(1024)(x)
x = Dense(512)(x)
x = Dense(256)(x)
x = Dense(123)(x)

out = x
model = Model(inp, out)
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
model.fit(data,y,epochs=1000,validation_split=0.2)
print(np.sum(LENLIST))