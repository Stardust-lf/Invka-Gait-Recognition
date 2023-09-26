from generateDataset import gen_DatasetNP,gen_DatasetOU
from InverseKoopman import computeCoder
from train_matrix import comp_mat
from gen_matrix_set import getMatrixResult
import datetime

starttime = datetime.datetime.now()
for TAG in ['nm']:
    for angle in [90]:
        print('Computing on {0} {1}'.format(TAG,angle))
        #dataset = gen_DatasetNP(TAG,angle)
        #gen_DatasetOU
        computeCoder(TAG,angle)
        comp_mat(TAG,angle)
        #getMatrix_LS(TAG,angle)
        getMatrixResult(TAG,angle)
# endtime = datetime.datetime.now()
# print((endtime-starttime).seconds)

