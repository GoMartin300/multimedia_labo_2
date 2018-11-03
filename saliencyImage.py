import sys
import cv2
import numpy as np
from skimage.exposure import rescale_intensity
import convolutions
from matplotlib import pyplot as plt

kernelFactor = 9

def kernel(n):
    factor = 1/pow(n, 2)
    kernelArray = np.full((n, n), factor)
    # print(kernelArray)
    return kernelArray

def showPlot(frame):
    max = np.max(frame)
    # plt.hist(frame.ravel(),  12, [0, frame]); plt.show()

orgimg_list = []
for i in range(1, 2):
    orgimg_list.append(cv2.imread(str(sys.argv[i]), 0))
frame = orgimg_list[0]
# 64 pixels, originaly
widthFrame = len(frame[0])
heightFrame = len(frame)
scaleFrameX = widthFrame/64
scaleFrameY = int((64/widthFrame)*heightFrame)
cv2.imshow('frame', frame)
frame = cv2.resize(frame, (64, 64), interpolation=cv2.INTER_CUBIC)
# frame = convolutions.convolve(frame, np.abs(kernel(5)))

showPlot(np.zeros((5, 5)))

imageFourier = np.fft.fft2(frame)

# show imageFourier
showPlot(imageFourier)
cv2.imshow('fourier', np.abs(imageFourier))

# we now have the spectrum analyse
# wich needs to be logaritmic:
imageConvolve = convolutions.convolve(frame, np.abs(kernel(9)))
fourierConvolve = np.fft.fft2(imageConvolve)
showPlot(fourierConvolve)
cv2.imshow('averagedSpectrum', np.abs(fourierConvolve))


# averagedSpectrum = np.multiply(imageConvolve, logSpectrum)
# showPlot(averagedSpectrum)
spectralResidual = np.subtract(fourierConvolve, imageFourier)
showPlot(spectralResidual)
saliencyImage = np.fft.ifft2(spectralResidual)
showPlot(saliencyImage)
max = np.max(saliencyImage)
saliencyImage = cv2.resize(np.abs(saliencyImage), (widthFrame, heightFrame))
cv2.imshow('saliency', saliencyImage)
#saliencyImage = np.asarray(saliencyImage*255/max, dtype=np.uint8)
#cv2.filter2D(saliencyImage,-1,kernel(9))
#newSaliencyImage = cv2.resize(np.abs(saliencyImage), (widthFrame, heightFrame))
#frame = cv2.resize(frame,(widthFrame,heightFrame))
#cv2.imshow('newSaliency', np.concatenate((frame, np.abs(newSaliencyImage)), axis=1))

while True:
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break



#
# convolve v d foto & dan imageFourier - die convolve
# daarna een ifft2 ervan doen
#
# # saliencyImage = np.fft.ifft2(frame)
# # saliencyImage = np.abs(saliencyImage)
# # saliencyImage = np.scal
# cv2.imshow('saliency', np.abs(saliencyImage))