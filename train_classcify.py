import tensorflow as tf
from tensorflow import keras
import random
# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
import pickle

BATCH_SIZE = 3
SAMPLE_LEN = 3 #true batch size is BATCH_SIZE * SAMPLE_LEN

LENLIST = [21, 23, 25, 26, 18, 22, 27, 22, 23, 22, 28, 27, 22, 18, 24, 20, 17, 20, 18, 24, 21, 20, 27, 20, 23, 22, 25, 20, 19, 23, 23, 21, 27, 21, 25, 19, 24, 22, 26, 22, 18, 25, 27, 24, 19, 27, 25, 17, 17, 20, 20, 23, 22, 22, 18, 22, 20, 22, 29, 17, 21, 19, 17, 25, 20, 23, 23, 20, 20, 24, 21, 20, 18, 23, 24, 23, 21, 18, 24, 19, 29, 23, 22, 19, 21, 22, 22, 18, 22, 17, 27, 26, 22, 20, 21, 19, 19, 21, 23, 23, 28, 18, 20, 15, 22, 22, 22, 25, 30, 22, 22, 20, 21, 21, 23, 22, 28, 25, 23, 21, 21, 18, 20]

model = keras.Sequential([
    tf.keras.layers.Conv2D(filters=32, kernel_size=4, strides=(4, 4), padding='VALID', name='conv1'),
    tf.keras.layers.Conv2D(filters=32, kernel_size=4, strides=(4, 4), padding='VALID', name='conv2'),
    keras.layers.Flatten(input_shape=(256, 256)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10)
])
def load_variavle(filename):
  f=open(filename,'rb')
  r=pickle.load(f)
  f.close()
  return r

def train_step(model,optimizer):
    indexs = random.sample(range(123), BATCH_SIZE)
    data = []
    for index in indexs:
        cyclenum = random.sample(range(LENLIST[index]),SAMPLE_LEN)
        for num in cyclenum:
            data.append(load_variavle('kopMatrix/sample{0}_cycle{1}'.format(index,num)))
    with tf.GradientTape() as t:
        #temp = model.koopmanMatrics
        result = model(data)
        y_true = [[i] * SAMPLE_LEN for i in indexs]
        y_true = np.reshape(np.array(y_true),3*3)
        loss = tf.keras.losses.SparseCategoricalCrossentropy(y_true,result)
        #loss_avg = tf.reduce_mean(loss)
        variables = model.trainable_variables
        grads = t.gradient(loss,variables)

        optimizer.apply_gradients(zip(grads,variables))

        return loss

optimizer = tf.keras.optimizers.Adagrad(learning_rate=0.0002, epsilon=1e-4)

for _ in range(3000):
    train_step(model,optimizer)