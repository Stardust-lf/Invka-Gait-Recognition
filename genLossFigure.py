import numpy as np
import pickle
import matplotlib.pyplot as plt
import cv2

def load_variavle(filename):
  f=open(filename,'rb')
  r=pickle.load(f)
  f.close()
  return r

loss08 = np.array(load_variavle('lossSample4lr0.8'),dtype=float)[:,300]
loss02 = np.array(load_variavle('lossSample4lr0.2'),dtype=float)[:,:150]
loss01 = np.array(load_variavle('lossSample4lr0.1'),dtype=float)[:,:300]
loss005 = np.array(load_variavle('lossSample4lr0.05'),dtype=float)[:,:300]
loss001 = np.array(load_variavle('lossSample4lr0.01'),dtype=float)[:,:600]
plt.rc('font',family='Times New Roman')
fig,ax = plt.subplots(2,2,constrained_layout=True,
                      figsize=(6,6),
                      dpi=1200
                      )

ax[0][0].plot(loss001.T,label=['loss0','loss1','loss2'],lw=1)
ax[0][0].set_title('lr=0.01',fontsize=12)
ax[0][0].tick_params(labelsize=12)
ax[0][0].legend(fontsize=10)
ax[0][0].grid()


ax[0][1].plot(loss005.T,label=['loss0','loss1','loss2'],lw=1)
ax[0][1].set_title('lr=0.05',fontsize=12)
ax[0][1].tick_params(labelsize=12)
ax[0][1].legend(fontsize=10)
ax[0][1].grid()

ax[1][0].plot(loss01.T,label=['loss0','loss1','loss2'],lw=1)
ax[1][0].set_title('lr=0.10',fontsize=12)
ax[1][0].tick_params(labelsize=12)
ax[1][0].legend(fontsize=10)
ax[1][0].grid()

ax[1][1].plot(loss02.T,label=['loss0','loss1','loss2'],lw=1)
ax[1][1].set_title('lr=0.20',fontsize=12)
ax[1][1].tick_params(labelsize=12)
ax[1][1].legend(fontsize=10)
ax[1][1].grid()

# ax[0][1].set_xlabel('epoch',fontsize=12)
# ax[0][0].set_xlabel('epoch',fontsize=12)
ax[1][0].set_xlabel('epoch',fontsize=12)
ax[1][1].set_xlabel('epoch',fontsize=12)
# plt.plot(loss005.T,c='b')
# plt.plot(loss01.T)
# plt.plot(loss02.T)
# plt.plot(loss08.T)
plt.savefig('TrainMatrixLosses')

