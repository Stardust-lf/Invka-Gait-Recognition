import cv2
import numpy as np
import pickle
cap = cv2.VideoCapture('MOVIE-0001.wmv')
data = []
ret = True
while(cap.isOpened() and ret):
    ret,frame = cap.read()
    if ret:
        data.append(frame)

data = np.array(data,dtype=float)
print(data.shape)
with open('heartData','wb+') as f:
    pickle.dump(data,f)