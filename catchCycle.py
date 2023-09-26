from PIL import Image
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pickle

INDEX = 2
SIZE = 64
SPLIT_VALUE = 1
PATH = 'D:\\DataSets\\CASIA-B\\030\\030\\nm-05\\090'
MIN_CYCLE_LENGTH = 9
STANDER_CYCLE_LENGTH = 12

def load_variavle(filename):
  f=open(filename,'rb')
  r=pickle.load(f)
  f.close()
  return r

def getSimilarity(img1,img2):
    sim = np.logical_xor(img1,img2)
    return 1 - np.sum(sim)/SIZE**2

def getCenter(image):
    image = np.array(image)

    # 找到人的最小最大高度与宽度
    height_min = (image.sum(axis=1) != 0).argmax()
    height_max = ((image.sum(axis=1) != 0).cumsum()).argmax()
    width_min = (image.sum(axis=0) != 0).argmax()
    width_max = ((image.sum(axis=0) != 0).cumsum()).argmax()
    center = [(width_min + width_max) / 2, (height_min + height_max) / 2]
    return center

def cut(image):
    image = np.array(image)
    height_min = (image.sum(axis=1) != 0).argmax()
    height_max = ((image.sum(axis=1) != 0).cumsum()).argmax()
    width_min = (image.sum(axis=0) != 0).argmax()
    width_max = ((image.sum(axis=0) != 0).cumsum()).argmax()
    head_top = image[height_min, :].argmax()
    size = height_max - height_min
    temp = np.zeros((size, size))
    l1 = head_top - width_min
    r1 = width_max - head_top
    centroid = np.array([(width_max + width_min) / 2, (height_max + height_min) / 2], dtype='int')
    flag = False
    if size <= width_max - width_min or size // 2 < r1 or size // 2 < l1:
        flag = True
        return temp,centroid,flag

    temp[:, (size // 2 - l1):(size // 2 + r1)] = image[height_min:height_max, width_min:width_max]
    return temp, centroid, flag

def getImgs(path):
    imgList = []
    centers = []
    for root,dirs,files in os.walk(path):
        for file in files:
            img = Image.open(root + '//' + file)
            img, center, flag = cut(img)
            if(flag):
                continue
            centers.append(center)
            cutImg = cv2.resize(img, (SIZE, SIZE))
            #img = Image.fromarray(img)
            #cutImg = img.resize((SIZE,SIZE),Image.ANTIALIAS)
            cutImg = np.array(cutImg)
            imgList.append(np.array(cutImg))
    imgArr = np.array(imgList)
    centers = np.array(centers)
    return imgArr, centers


def getDistributed(imgList,stdPos):
    simillist = []
    centers = []
    for j in range(1,len(imgList)-1):
        simillist.append(getSimilarity(imgList[stdPos],imgList[j]))
    similArr = np.array(simillist)
    return similArr

def getAllDistributed(imgList):
    similarities = []
    for i in range(0,len(imgList)-1):
        similArr = getDistributed(imgList,i)
        similarities.append(similArr)
    return similarities

def getCutPos(semilArr):
    cutPos = []
    for i in range(1,len(semilArr)-1):
        if(semilArr[i] >= semilArr[i-1] and semilArr[i] >= semilArr[i+1]):
            cutPos.append([i,semilArr[i]])
    return cutPos

def rmNearValue(cutPos):
    j = 0
    while j < len(cutPos) - 1:
        if cutPos[j + 1][0] - cutPos[j][0] <= MIN_CYCLE_LENGTH:
            if cutPos[j][1] < cutPos[j+1][1]:
                cutPos.pop(j)
                j -= 1
            else:
                cutPos.pop(j+1)
                j -= 1
        j += 1
    return cutPos

def catchCycle(path):
    imgList, centers = getImgs(path)
    similarities = getAllDistributed(imgList)
    variances = []
    for item in similarities:
        variances.append(np.var(item))
    bestStd = np.argmax(variances)
    semilArr = similarities[bestStd]
    cutPos = getCutPos(semilArr)
    cutPos = rmNearValue(cutPos)
    return semilArr,cutPos,imgList, centers

def catchCycleForArray(array):

    similarities = getAllDistributed(array)
    variances = []
    for item in similarities:
        variances.append(np.var(item))
    bestStd = np.argmax(variances)
    semilArr = similarities[bestStd]
    cutPos = getCutPos(semilArr)
    cutPos = rmNearValue(cutPos)
    return semilArr,cutPos

def sepCycle(path):
    count = 0
    for root,dirs,files in os.walk(path):
        for each in files:
            count+=1
    if count <= STANDER_CYCLE_LENGTH:
        print('Folder has {0} files, has been jumped'.format(count))
        raise OSError('Folder has {0} files, has been jumped'.format(count))
    semilArr,cutPos,imgList, centers = catchCycle(path)
    # plt.plot(semilArr)
    # plt.show()
    if(len(cutPos)<2):
        return [],[]
    cutPos = [i[0] for i in cutPos]
    #print(cutPos)
    imgSlipList = []
    centersList = []
    for i in range(0,len(cutPos) - 1):
        if(cutPos[i] + STANDER_CYCLE_LENGTH<=len(imgList)):
            imgSlipList.append(imgList[cutPos[i] - 1:cutPos[i] + STANDER_CYCLE_LENGTH - 1])
            centersList.append(centers[cutPos[i] - 1:cutPos[i] + STANDER_CYCLE_LENGTH - 1])
        else:
            imgSlipList.append(imgList[cutPos[i] - 1:cutPos[i + 1] - 1])
            centersList.append(centers[cutPos[i] - 1:cutPos[i + 1] - 1])
            while(len(imgSlipList[-1])!=STANDER_CYCLE_LENGTH):
                if(len(imgSlipList[-1]) > STANDER_CYCLE_LENGTH):
                    imgSlipList[-1] = imgSlipList[-1][:-1]
                else:
                    imgSlipList.append(imgSlipList[-1][-1])
    imgSlipArr = np.array(imgSlipList,dtype=np.float)
    centersArr = np.array(centersList,dtype=list)
    #imgSlipArr = fixImg(imgSlipArr)
    for i in range(len(imgSlipArr)):

        avgE = np.average(imgSlipArr)
        res = False
        for fig in imgSlipArr[i]:
            if np.average(fig) > avgE + 5:
                res = True
                break
        if res:
            imgSlipArr = np.delete(imgSlipArr,i,axis=0)
            break
        imgSlipArr[i] = rearrange(imgSlipArr[i])
    return imgSlipArr,centersArr

#分割图片
def sepCycle_OU(path):
    STANDER_CYCLE_LENGTH = 24
    count = 0
    for root,dirs,files in os.walk(path):
        for each in files:
            count+=1
    if count <= STANDER_CYCLE_LENGTH:
        print('Folder has {0} files, has been jumped'.format(count))
        raise OSError('Folder has {0} files, has been jumped'.format(count))
    semilArr,cutPos,imgList, centers = catchCycle(path)
    # plt.plot(semilArr)
    # plt.show()
    if(len(cutPos)<2):
        return [],[]
    cutPos = [i[0] for i in cutPos]
    # print(cutPos)
    # plt.plot(semilArr)
    # plt.vlines(x=cutPos,ymin=0,ymax=1)
    # plt.show()
    imgSlipList = []
    centersList = []
    for i in range(0,len(cutPos) - 1):
        if(cutPos[i] + STANDER_CYCLE_LENGTH<=len(imgList)):
            imgSlipList.append(imgList[cutPos[i] - 1:cutPos[i] + STANDER_CYCLE_LENGTH - 1])
            centersList.append(centers[cutPos[i] - 1:cutPos[i] + STANDER_CYCLE_LENGTH - 1])
        else:
            imgSlipList.append(imgList[cutPos[i] - 1:cutPos[i + 1] - 1])
            centersList.append(centers[cutPos[i] - 1:cutPos[i + 1] - 1])
            while(len(imgSlipList[-1])!=STANDER_CYCLE_LENGTH):
                if(len(imgSlipList[-1]) > STANDER_CYCLE_LENGTH):
                    imgSlipList[-1] = imgSlipList[-1][:-1]
                else:
                    imgSlipList.append(imgSlipList[-1][-1])
    imgSlipArr = np.array(imgSlipList,dtype=np.float)
    centersArr = np.array(centersList,dtype=list)
    #imgSlipArr = fixImg(imgSlipArr)
    for i in range(len(imgSlipArr)):

        avgE = np.average(imgSlipArr)
        res = False
        for fig in imgSlipArr[i]:
            if np.average(fig) > avgE + 5:
                res = True
                break
        if res:
            imgSlipArr = np.delete(imgSlipArr,i,axis=0)
            break
        imgSlipArr[i] = rearrange(imgSlipArr[i])
    return imgSlipArr,centersArr

def sepCycleArray(array):
    STANDER_CYCLE_LENGTH = 18
    count = len(array)
    if count <= STANDER_CYCLE_LENGTH:
        print('Folder has {0} files, has been jumped'.format(count))
        raise OSError('Folder has {0} files, has been jumped'.format(count))
    semilArr,cutPos = catchCycleForArray(array)
    if(len(cutPos)<2):
        return [],[]
    cutPos = [i[0] for i in cutPos]
    #print(cutPos)
    imgSlipList = []
    centersList = []
    for i in range(0,len(cutPos) - 1):
        if(cutPos[i] + STANDER_CYCLE_LENGTH<=count):
            imgSlipList.append(array[cutPos[i] - 1:cutPos[i] + STANDER_CYCLE_LENGTH - 1])
        else:
            imgSlipList.append(array[cutPos[i] - 1:cutPos[i + 1] - 1])
            while(len(imgSlipList[-1])!=STANDER_CYCLE_LENGTH):
                if(len(imgSlipList[-1]) > STANDER_CYCLE_LENGTH):
                    imgSlipList[-1] = imgSlipList[-1][:-1]
                else:
                    imgSlipList.append(imgSlipList[-1][-1])
    imgSlipArr = np.array(imgSlipList,dtype=np.float)
    centersArr = np.array(centersList,dtype=list)
    #imgSlipArr = fixImg(imgSlipArr)
    for i in range(len(imgSlipArr)):

        avgE = np.average(imgSlipArr)
        res = False
        for fig in imgSlipArr[i]:
            if np.average(fig) > avgE + 5:
                res = True
                break
        if res:
            imgSlipArr = np.delete(imgSlipArr,i,axis=0)
            break
        imgSlipArr[i] = rearrange(imgSlipArr[i])
    return imgSlipArr,centersArr

def sepAllCycle(count):
    allCycle = []
    for i in range(1,count+1):
        cycleTemp = []
        for j in range(1,7):
            path = 'D:\\DataSets\\CASIA-B\\{0}\\{0}\\nm-0{1}\\090\\'.format(str(i).zfill(3),str(j))
            cycle,center = sepCycle(path)
            cycleTemp.append(cycle)
        cycleArr = cycleTemp[0]
        for i in range(1,len(cycleTemp)):
            if(type(cycleArr) == np.ndarray and type(cycleTemp[i] == np.ndarray)):
                cycleArr = np.concatenate((cycleArr,cycleTemp[i]),axis=0)
        allCycle.append(cycleArr)
    return allCycle


def getHeatMaps(path) -> list:
    cycle, centers = sepCycle(path)
    headMapList = []
    for cyc in cycle:
        imgSum = np.zeros([int(SIZE * SPLIT_VALUE), SIZE])
        for item in cyc:
            item = item[SIZE - int(SIZE * SPLIT_VALUE):, :]
            item = item / STANDER_CYCLE_LENGTH
            imgSum = np.add(imgSum,item)
        headMapList.append(imgSum)
    return headMapList

def getDiffOfIndex(data,index):
    diff = []
    for i in range(len(data)-index):
        diff.append(1-getSimilarity(data[i],data[i+index]))
    diff = np.array(diff)
    diff *= SIZE**2*SPLIT_VALUE
    return diff

def rearrange(cycle):
    varList = []
    for i in range(STANDER_CYCLE_LENGTH):
        varList.append(np.var(cycle[i]))
    varArray = np.array(varList)
    varMaxPos = varArray.argmax()
    cycle = np.roll(cycle,-varMaxPos,axis=0)
    return cycle

def fixImg(cycles):
    avgImg = np.average(cycles,0)
    splitPos = int(avgImg.shape[1] * 0.6)
    deltaImg = avgImg[:, :splitPos, :]
    # deltaImg[deltaImg>85] = 255
    # deltaImg[deltaImg<=85] = 0
    cycles[:,:,:splitPos,:] = (cycles[:,:,:splitPos,:] + deltaImg)/2
    cycles[cycles>70] = 255
    cycles[cycles<=70] = 0
    cycles = cycles * (255/cycles.max())
    # for j in range(len(cycles)):
    #     for i in range(len(cycles[j])):
    #         cycles[j][i] = np.array(cycles[j][i],dtype=np.float32)
    #         cycles[j][i] = cv2.medianBlur(cycles[j][i],5)

    return cycles


# def getPerfectCycles(count):
#     data = sepAllCycle(count) #[类，样本，段，张，宽，高]
#     for aClass in data:




if __name__ == '__main__':
    semilArr,cutPos, imgList, centers= catchCycle(PATH)
    plt.plot(semilArr)
    for item in cutPos:
        plt.plot((item[0],item[0]),(0.85,0.98))
    plt.show()


    data, centers = sepCycle(PATH)
    imgSum = getHeatMaps(PATH)[0]

    img = Image.fromarray(np.uint8(imgSum))
    img.show()
    speedSquare = []
    aData = data[INDEX]
    center = centers[INDEX]
    bData = []
    for img in aData:
        bData.append(img[SIZE - int(SIZE * SPLIT_VALUE):, :])

    aData = np.array(bData,dtype=int)
    for i in range(len(center)-1):
        speedSquare.append((center[i][0] - center[i+1][0]) **2 + (center[i][1] - center[i+1][1]) ** 2)
    speed = np.array(speedSquare)
    diff = getDiffOfIndex(aData, 1)
    diff2 = getDiffOfIndex(aData, 2)
    diff3 = getDiffOfIndex(aData, 3)

    plt.plot(diff)
    plt.plot(diff2)
    plt.plot(diff3)

    plt.plot(speed)
    plt.show()


