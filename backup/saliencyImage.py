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
    plt.hist(frame.ravel(),  12, [0, 120]); plt.show()

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
logImageFourier = np.log(imageFourier)
showPlot(imageFourier)
cv2.imshow('fourier', np.abs(imageFourier))

# we now have the spectrum analyse
# wich needs to be logaritmic:
imageConvolve = convolutions.convolve(frame, np.abs(kernel(9)))
cv2.imshow('org', np.abs(frame))
cv2.imshow('blur', np.abs(imageConvolve))
fourierConvolve = np.fft.fft2(imageConvolve)
logFourierConvolve = np.log(fourierConvolve)
showPlot(fourierConvolve)
cv2.imshow('averagedSpectrum', np.abs(fourierConvolve))
# averagedSpectrum = np.multiply(imageConvolve, logSpectrum)
# showPlot(averagedSpectrum)
phase = np.angle(imageFourier)
spectralResidual = np.subtract(logFourierConvolve, logImageFourier)
spectralResidual = np.exp(spectralResidual+1j*phase)**2

showPlot(spectralResidual)

saliencyImage = np.fft.ifft2(spectralResidual)
showPlot(saliencyImage)
# saliencyImage = rescale_intensity(np.abs(saliencyImage), in_range=(0, 255))
newSaliencyImage = cv2.resize(np.abs(saliencyImage), (widthFrame, heightFrame))
cv2.imshow('saliency', newSaliencyImage)

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