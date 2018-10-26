import sys
import cv2
import numpy as np
from skimage.exposure import rescale_intensity
import convolutions
from matplotlib import pyplot as plt

cap = cv2.VideoCapture(str(sys.argv[1]))

ret, frame = cap.read()



def kernel(n):
    factor = 1/pow(n, 2)
    kernelArray = np.full((n, n), factor)
    # print(kernelArray)
    return kernelArray

def imageConvolution(frame,kernel):
    """
    1/ place the frame in separate matrices
    2/ perform the convolution on a frame
    3/ resize the frame
    :param frame: an image of a video
    :param kernel: is varying
    :return: return the convolution image
    """
    widthReturnArray = int(len(frame[0])-len(kernel)+1)
    heightReturnArray = int(len(frame)-len(kernel)+1)
    widthReturnArray = len(frame[0])
    heightReturnArray = len(frame)

    returnArray = np.zeros((heightReturnArray, widthReturnArray))
    x = 0
    y = 0

    kernelWidth = len(kernel[0])

    while y <= (len(frame)-len(kernel)):
        while x <= (len(frame[0]) - len(kernel[0])):
            # imaginary array from x to x+kernelwidth en from y to y+kernelheight
            currentSum = 0
            kernelX = 0
            kernelY = 0
            for b in range(0, kernelWidth):

                for a in range(0, kernelWidth):
                    currentSum = currentSum + frame[b+y][a+x]*kernel[b][a]
            returnArray[y][x] = currentSum
            # print(currentSum)
            x = x+1
        x = 0
        y = y+1
    # print(returnArray)
    # get the size of kernel and take a matrice from the frame

    """
    size for width & height:
    a = frameSize%kernelSize
    => framSize - a  / kernelSize
    """
    #min = np.min(returnArray)
    #np.sum(returnArray-min)
    # rescale
    maxValue = np.max(returnArray)
    factor = 255/maxValue
    returnArray = np.multiply(returnArray, factor)
    # returnArray = rescale_intensity(returnArray, in_range=(0, 255), np.max(returnArray))

    return returnArray

def showPlot(frame):
    plt.hist(frame.ravel(),  12, [0, 120]); plt.show()


while cap.isOpened():
    orgValuePixels = []
    ret, frame = cap.read()
    frame = cv2.resize(frame, (0, 0), fx=0.3, fy=0.3)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imshow('origineel', frame)
    cv2.waitKey(1)
    imageFourier = np.fft.fft(frame)
    # fourrierHist = np.histogram(imageFourier.ravel(), 256, [0, 256])
    showPlot(imageFourier)


    # we now have the spectrum analyse
    # wich needs to be logaritmic:
    logSpectrum = np.log(imageFourier)
    showPlot(logSpectrum)
    """
    kernl = kernel(9)
    # kernl = np.uint8(kernl)
    #imageConvolve = convolutions.convolve(imageFourier.real, kernl)
    laplacian = np.array((
        [1/9, 1/9, 1/9],
        [1/9, 1/9, 1/9],
        [1/9, 1/9, 1/9]), dtype="int")

    sharpen = np.array((
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]), dtype="int")
    """
    imageConvolve = convolutions.convolve(frame, np.real(kernel(9)))
    showPlot(imageConvolve)
    # imageConvolve = np.convolve(frame, np.real(kernel(9)))
    # imageConvolve = imageConvolution(frame, laplacian)

    averagedSpectrum = np.multiply(imageConvolve, logSpectrum)
    showPlot(averagedSpectrum)

    spectralResidual = np.subtract(logSpectrum, averagedSpectrum)
    showPlot(spectralResidual)
    # imageSubtract = np.zeros(len(imageConvolve[0], len(imageConvolve)))
    # for y in range(0, len(imageSubtract)):
    #     for x in range(0, len(imageSubtract[0])):
    #         imageSubtract[y][x] = imageFourier[y][x]-imageConvolve[y][x]
    saliencyImage = np.fft.ifft2(spectralResidual)
    # saliencyImage = rescale_intensity(np.real(saliencyImage), in_range=(0, 255))
    cv2.imshow('saliency', np.real(saliencyImage))




    #
    # convolve v d foto & dan imageFourier - die convolve
    # daarna een ifft2 ervan doen
    #
    # # saliencyImage = np.fft.ifft2(frame)
    # # saliencyImage = np.real(saliencyImage)
    # # saliencyImage = np.scal
    # cv2.imshow('saliency', np.real(saliencyImage))

