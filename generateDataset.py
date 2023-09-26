import numpy as np
from catchCycle import sepCycle,sepCycle_OU
import pickle
import os
def gen_DatasetNP(tag,angle):
    dataSet = []
    for i in range(1,125):
        if i==94:continue
        people = []
        for j in range(1,3):
            path = 'D:\\DataSets\\CASIA-B\\{0}\\{0}\\{1}-0{2}\\{3}'.format(str(i).zfill(3),tag,str(j),str(angle).zfill(3))
            try:
                cycles = sepCycle(path)[0]
            except OSError:
                continue
            people+=list(cycles)
        people = np.array(people)
        if(len(people)>3):
            dataSet.append(people)
    with open('data124_{0}_{1}'.format(tag,angle),'wb+') as f:
        pickle.dump(dataSet,f)
    return dataSet

def gen_DatasetOU(tag,angle):
    dataSet = []
    n = 0
    for root, dirs, files in os.walk("E:\OU_MVLP\Silhouette_090-00", topdown=False):
        for dirname in dirs:
            if n >= 300:
                break
            print(os.path.join(root, dirname))
            path = os.path.join(root, dirname)
            try:
                cycles = sepCycle_OU(path)[0]
                cycles = np.reshape(cycles,[-1,12,2,64,64])
                cycles = cycles.transpose([0,2,1,3,4])
                cycles = np.reshape(cycles,[-1,12,64,64])
                dataSet.append(np.array(cycles, dtype=np.float))

                print(cycles.shape)
                print(len(dataSet))
                n+=1
            except Exception as err:
                continue
    with open('data124_{0}_{1}'.format(tag,angle),'wb+') as f:
        pickle.dump(dataSet,f)
    return dataSet

if __name__ == '__main__':
    gen_DatasetOU('nm',90)
