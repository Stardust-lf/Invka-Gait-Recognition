import tensorflow as tf


def encoder(input,filter):
    img = tf.nn.conv2d(input,filter[0],strides=1,padding=2)
    img = tf.nn.conv2d(img,filter[1],strides=2,padding=1)
    img = tf.nn.conv2d(img,filter[2],strides=1,padding=1)
    img = tf.nn.conv2d(img,filter[3],strides=2,padding=1)
    img = tf.nn.conv2d(img,filter[4],strides=1,padding=1)
    img = tf.nn.conv2d(img,filter[5],strides=2,padding=1)
    return img

def decoder(input,filter):
    img = tf.nn.conv2d_transpose(input,filter[0],strides=2,padding=1)
    img = tf.nn.conv2d_transpose(img, filter[1], strides=1, padding=1)
    img = tf.nn.conv2d_transpose(img, filter[2], strides=2, padding=1)
    img = tf.nn.conv2d_transpose(img, filter[3], strides=1, padding=1)
    img = tf.nn.conv2d_transpose(img, filter[4], strides=2, padding=1)
    img = tf.nn.conv2d_transpose(img, filter[5], strides=1, padding=2)
    return img

class koopManTrain(tf.keras.Model):
    def __init__(self, filters, kmMatrix):
        super(koopManTrain, self).__init__()
        self.filters = filters
        self.kmMatrix = kmMatrix

        self.dense1 = tf.keras.layers.Dense(units = 1024)
        self.dense2 = tf.keras.layers.Dense(units = 128)

        self.dense3 = tf.keras.layers.Dence(units = 1024)
        self.dense4 = tf.keras.layers.Dense(units = 2048)

        self.dense5 = tf.keras.layers.Dense(units=1024)
        self.dense6 = tf.keras.layers.Dense(units=128)
        self.dense7 = tf.keras.layers.Dence(units = 25)

    def call(self, inputs):
        print(inputs.shape)
        cycle = inputs[0]
        img = cycle[0]
        z = encoder(img,self.filters[0:6])
        z_preNext = tf.matmul(z,self.kmMatrics[0])
        x = decoder(z, self.filters[6:12])
        x_preNext = decoder(z_preNext, self.filters[6:12])






