import tensorflow as tf
import catchCycle
import matplotlib.pyplot as plt
import numpy as np
import pickle
from InverseKoopman import MyModel,preprocess

STD_LEN = 12
KOOPMAN_SIZE = 2048
NUM_EPOCHS = 300000
NUM_DENSE_UNITS  = 2048

if __name__ == '__main__':
    model = MyModel(tf.random_normal_initializer(mean=1., stddev=2.))
    data = catchCycle.load_variavle('data124')
    data = preprocess(data)
    data[4] = data[5]
    # print(data[[0,1,2]].shape)
    model([data[0]],[0])
    model.load_weights('coder_fin_nm')
    x,phiX,dephiPhiX,kPhix,dePhiKPhiX,x_next,phiX_next = model([data[0]],[0])

    x_np = tf.squeeze(x)
    x_np = x_np.numpy()
    dephiPhiX_np = tf.squeeze(dephiPhiX)
    dephiPhiX_np = dephiPhiX_np.numpy()
    phiX_np = tf.squeeze(phiX)
    phiX_np = phiX_np.numpy()
    with open('phiX_test','wb+') as f:
        pickle.dump(phiX_np,f)
    dePhiKPhiX_np = tf.squeeze(dePhiKPhiX).numpy()
    x_next_np = tf.squeeze(x_next).numpy()
    plt.imshow(dePhiKPhiX.numpy()[0][0])
    plt.savefig('dePhiKPhiX')
    plt.imshow(x_next.numpy()[0][0])
    plt.savefig('x_next')
    file = open('dePhiKPhiX','wb')
    pickle.dump(dePhiKPhiX.numpy()[0][0],file)
    # fig,ax = plt.subplots(2,3,sharex='all',sharey='all')
    # aix = ax[0, 0]
    # aix.set_title('x')
    # aix.imshow(x.numpy()[0][0])
    #
    # aix = ax[0, 1]
    # aix.set_title('phiX')
    # aix.imshow(phiX.numpy()[0][0])
    #
    # aix = ax[0, 2]
    # aix.set_title('dephiPhiX')
    # aix.imshow(dephiPhiX.numpy()[0][0])
    #
    # aix = ax[1, 0]
    # aix.set_title('kPhix')
    # aix.imshow(kPhix.numpy()[0][0])
    #
    # aix = ax[1, 1]
    # aix.set_title('dePhiKPhiX')
    # aix.imshow(dePhiKPhiX.numpy()[0][0])
    #
    # aix = ax[1, 2]
    # aix.set_title('x_next')
    # aix.imshow(x_next.numpy()[0][0])
    #
    # plt.show()
    #
    #
    # for i in range(len(x_np)):
    #     print(x_np[i].shape)
    #     loss = np.linalg.norm(x_np[i] - dephiPhiX_np[i])
    #     print('loss0{0}: {1}'.format(i,loss))
    #
    # file = open('Koopman_Matrixs','wb')
    # pickle.dump(model.koopmanMatrics,file)

