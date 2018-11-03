# https://www.mathworks.com/matlabcentral/mlc-downloads/downloads/submissions/38512/versions/1/previews/CLE/saltool/SpectralR/SpectralResidualSaliency.m/index.html?access_key=

import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('paarden.jpg',0)
width=len(img[0])
height=len(img)

img = cv2.resize(img, (64, 64))
cv2.imshow("img", img)

kernel = np.ones((3,3),np.float32)/9

f = np.fft.fft2(img)
logamp = np.log(np.abs(f))
phase = np.angle(f);

sr = logamp -  cv2.filter2D(logamp,-1,kernel)

print(np.abs(np.fft.ifft2(np.exp(sr + 1j*phase))))
sm = np.abs(np.fft.ifft2(np.exp(sr + 1j*phase)))**2

msm = np.max(sm)
sm = np.asarray(sm*255/msm, dtype=np.uint8)
cv2.filter2D(sm,-1,kernel)
sm = cv2.resize(sm, (height, width))
img = cv2.resize(img, (height, width))
cv2.imshow("sm", np.concatenate((img, sm), axis=1))
cv2.imshow("new",np.saliency)
while(True):
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
