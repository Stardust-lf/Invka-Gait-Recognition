from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout, Conv3D, MaxPooling3D
import pickle
from tensorflow_addons import losses
import numpy as np
import tensorflow as tf
from tensorflow import optimizers
import matplotlib.pyplot as plt
import gc
lr=0.000320

y_true = []
for i in range(123):
  y_true += [i]*10

def load_variavle(filename):
  f=open(filename,'rb')
  r=pickle.load(f)
  f.close()
  return r

data = load_variavle('data124')
data = np.delete(data,4)
inpOri = [data[i][:10] for i in range(len(data))]
for item in inpOri:
  print(item.shape)
inpOri = np.array(inpOri,dtype=np.float)
inpOri/=255
print(inpOri.shape)
inpOri = np.expand_dims(inpOri,-1)
squirrelInp = load_variavle('squirrelCycle')
squirrelInp = np.reshape(squirrelInp,[1,12,64,64,1])
squirrelInp = squirrelInp/255
print(squirrelInp.shape,'squirrel')
inpOri = np.reshape(inpOri,[-1,12,64,64,1])

inpOri = tf.convert_to_tensor(inpOri,dtype=tf.float32)
inp = Input((12, 64, 64, 1))
x1 = Conv3D(16, kernel_size=(3, 3, 3), activation="relu",padding='Same')(inp)
x21 = Conv3D(16, kernel_size=(1, 3, 3), activation="relu",padding='Same')(inp)
x22 = Conv3D(64, kernel_size=(3, 1, 1), activation="relu",padding='Same')(x21)
x23 = Conv3D(16, kernel_size=(1, 1, 1), activation="relu",padding='Same')(x22)
x = x1+x23
x = Flatten()(x)
x = Dense(units=128)(x)
x = Dense(units=128)(x)
out = Dense(units=123)(x)

optimizer = optimizers.Adam(learning_rate=0.0005, epsilon=1e-4)
model = Model(inp, out)
model.compile(metrics=['accuracy'],optimizer=optimizer,loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True))
history = model.fit(inpOri,y=np.array(y_true,dtype=np.int),batch_size=8,epochs=20,shuffle=True,validation_split=0.8)
with open('3DLoss00005','wb+') as f:

  pickle.dump([history.history['loss'],history.history['val_loss']],f)

fig1, ax_acc = plt.subplots()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Model - Loss')
plt.legend(['Training', 'Validation'], loc='lower right')
plt.show()