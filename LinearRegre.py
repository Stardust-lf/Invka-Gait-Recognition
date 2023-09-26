import numpy as np
import pickle
import matplotlib.pyplot as plt

def load_variavle(filename):
  f=open(filename,'rb')
  r=pickle.load(f)
  f.close()
  return r

def getK(cycle):
    oriMat = cycle[:11]
    outMat = cycle[1:]
    #print(oriMat.shape)
    unitMat = np.zeros(shape=[64,64])
    solveMat = np.zeros(shape=[4096,4096])
    solveRes = np.zeros(shape=[4096])
    for i in range(64):
        for j in range(64):
            Y = np.multiply(oriMat[:,i],oriMat[:,j])
            unitMat[i][j] = np.sum(np.reshape(Y,(Y.size,)))
    print('UnitMat created')
    for i in range(0,4095,64):
        solveMat[i:i+64,i:i+64] = unitMat
    for i in range(0,64):
        for j in range(64):
            Y = np.multiply(oriMat[:,j],outMat[:,i])
            solveRes[i*64+j] += np.sum(np.reshape(Y,(Y.size,)))
    print('Solving')
    try:
        res = np.linalg.solve(solveMat,solveRes)
        res = np.reshape(res,[64,64])
        return res
    except Exception as err:
        print("LinAlgError! Jump!")
        return np.zeros(shape=[64,64])


if __name__ == '__main__':
    data = load_variavle('phiX_test')
    data = data[0]


    K = getK(data)
    result = []
    for item in data[:11]:
        result.append(np.matmul(K,item))
    y_pred = np.array(result)
    Y = data[1:]
    diff = Y-y_pred
    diff = np.multiply(diff,diff)
    print(y_pred.shape)
    print(np.sum(np.reshape(diff,(diff.size,))))
    # plt.imshow(K)
    # plt.show()


