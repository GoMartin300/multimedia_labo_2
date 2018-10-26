# https://www.mathworks.com/matlabcentral/mlc-downloads/downloads/submissions/38512/versions/1/previews/CLE/saltool/SpectralR/SpectralResidualSaliency.m/index.html?access_key=
import sys
import cv2
import convolutions
import numpy as np
from matplotlib import pyplot as plt

def kernel(n):
    factor = 1/pow(n, 2)
    kernelArray = np.full((n, n), factor)
    # print(kernelArray)
    return kernelArray

def showPlot(frame):
    maxFrame = np.max(frame)
    plt.hist(frame.ravel(),  12, [0, maxFrame]); plt.show()

img = cv2.imread(str(sys.argv[1]), 0)
cv2.imshow('original', img)
width=img.shape[0]
height=img.shape[1]

img = cv2.resize(img, (64, 64))
cv2.imshow("img", img)

kernel = kernel(9)

f = np.fft.fft2(img)
logamp = np.log(np.abs(f))
phase = np.angle(f)

sr = logamp -  cv2.filter2D(logamp,-1,kernel)
showPlot(sr)
print(np.abs(np.fft.ifft2(np.exp(sr + 1j*phase))))
sm = np.abs(np.fft.ifft2(np.exp(sr + 1j*phase)))**2

msm = np.max(sm)
sm = np.asarray(sm*255/msm, dtype=np.uint8)
cv2.filter2D(sm,-1,kernel)
sm = cv2.resize(sm, (height, width))
img = cv2.resize(img, (height, width))
cv2.imshow("sm", np.concatenate((img, sm), axis=1))

while(True):
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
