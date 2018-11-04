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
    maxFrame = np.max(frame)
    plt.hist(frame.ravel(),  12, [0, maxFrame]); plt.show()

def getSaliencyFrame(frame):
    widthFrame = len(frame[0])
    heightFrame = len(frame)
    cv2.imshow('frame', frame)
    #compress the frame
    frame = cv2.resize(frame, (64, 64), interpolation=cv2.INTER_CUBIC)
     # fourier analyse
    imageFourier = np.fft.fft2(frame)
    # show imageFourier
    showPlot(imageFourier)
    cv2.imshow('fourier', np.abs(imageFourier))
    # we now have the spectrum analyse
    # by using convolve we can blur the frame
    imageConvolve = convolutions.convolve(frame, np.abs(kernel(9)))
    # take the fourier analyse of the convolve
    fourierConvolve = np.fft.fft2(imageConvolve)
    showPlot(fourierConvolve)
    cv2.imshow('averagedSpectrum', np.abs(fourierConvolve))
    #substract two images
    spectralResidual = np.subtract(fourierConvolve, imageFourier)
    showPlot(spectralResidual)
    # inverse fourier analyse
    saliencyImage = np.fft.ifft2(spectralResidual)
    showPlot(saliencyImage)
    saliencyImage = cv2.resize(np.abs(saliencyImage), (widthFrame, heightFrame))
    cv2.imshow('saliency', saliencyImage)

    # while True:
    #     if cv2.waitKey(1) & 0xFF == ord('q'):
    #         break
    return saliencyImage


