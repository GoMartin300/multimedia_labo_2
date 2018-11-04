# https://www.mathworks.com/matlabcentral/mlc-downloads/downloads/submissions/38512/versions/1/previews/CLE/saltool/SpectralR/SpectralResidualSaliency.m/index.html?access_key=
import sys
import cv2
import convolutions
import numpy as np
from matplotlib import pyplot as plt
from saliencyImage import kernel
from saliencyImage import getSaliencyFrame

def getSaliency(img):
    return cv2Methode(img)

def cv2Methode(img):
    saliency = cv2.saliency.StaticSaliencySpectralResidual_create()
    (success, saliencyMap) = saliency.computeSaliency(img)
    saliencyMap = (saliencyMap * 255).astype("uint8")
    return saliencyMap

# reformed version from minerva
def residualMethode(img):
    width=img.shape[0]
    height=img.shape[1]
    img = cv2.resize(img, (64, 64))
    kernl = kernel(5)
    f = np.fft.fft2(img)
    logamp = np.log(np.abs(f))
    phase = np.angle(f)
    sr = logamp - cv2.filter2D(logamp, -1, kernl)
    sm = np.abs(np.fft.ifft2(np.exp(sr + 1j*phase)))**2
    msm = np.max(sm)
    sm = np.asarray(sm*255/msm, dtype=np.uint8)
    cv2.filter2D(sm, -1, kernl)

    return sm

# method based of a paper
def residualMethode2(img):
    return getSaliencyFrame(img)
