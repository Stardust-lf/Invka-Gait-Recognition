import tensorflow as tf
import catchCycle

PATH = 'D://DataSets//CASIA-B//{0}//{0}//nm-01/090'.format('005')
STD_LEN = 12
KOOPMAN_SIZE = 20

data, center = catchCycle.getImgs(PATH)


print(data.shape)

class MyModel(tf.keras.Model):
    def __init__(self,initializer,index):
        super(MyModel, self).__init__()
        self.index = index
        self.koopmanMatrics = tf.Variable(initializer(shape=[STD_LEN-1,KOOPMAN_SIZE,KOOPMAN_SIZE],dtype=tf.float32))
        # self.conv1 = tf.keras.layers.conv2D(filters=1,kernel_size=3, strides= (1, 1), padding='VALUE')
        # self.conv2 = tf.keras.layers.conv2D(filters=1, kernel_size=3, strides=(1, 1), padding='VALUE')
        # self.conv3 = tf.keras.layers.conv2D(filters=1, kernel_size=3, strides=(1, 1), padding='VALUE')
        self.flatten1 = tf.keras.layers.Flatten(input_shape=(1, 44, 64))
        self.dense1 = tf.keras.layers.Dense(units=20,name='dense1')
        self.dense2 = tf.keras.layers.Dense(units=20,name='dense2')
        self.dense3 = tf.keras.layers.Dense(units=20,name='dense3')
        self.dense4 = tf.keras.layers.Dense(units=20,name='dense4')
        self.dense5 = tf.keras.layers.Dense(units=20,name='dense5')
        self.generatorY = tf.keras.layers.Dense(units=2,name='generatorY')

        self.undense1 = tf.keras.layers.Dense(units=20,name='undense1')
        self.undense2 = tf.keras.layers.Dense(units=20,name='undense2')
        self.undense3 = tf.keras.layers.Dense(units=20,name='undense3')
        self.undense4 = tf.keras.layers.Dense(units=20,name='undense4')
        self.undense5 = tf.keras.layers.Dense(units=20,name='undense5')
        self.regeneratorX = tf.keras.layers.Dense(units=1024)
        initializer = tf.random_normal_initializer(mean=1., stddev=2.)
        self.K = tf.Variable(initializer(shape=[30,20,20],dtype=tf.float32))


    def encoder(self,x):
        data = self.flatten1(x)
        data = self.dense1(data)
        data = self.dense2(data)
        data = self.dense3(data)
        data = self.dense4(data)
        data = self.dense5(data)
        y = self.generatorY(data)
        return y

    def decoder(self,y):
        data = self.undense1(y)
        data = self.undense2(data)
        data = self.undense3(data)
        data = self.undense4(data)
        data = self.undense5(data)
        x = self.regeneratorX(data)
        return x

    def __call__(self, inputs):
        x = inputs
        y = self.encoder(x)
        reX = self.decoder(y)
        y_next = tf.matmul(y,self.K[self.index])
        x_next = self.decoder(y_next)

        return reX,x_next






