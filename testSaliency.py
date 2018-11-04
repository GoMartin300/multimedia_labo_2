from spectralresidualsaliency import getSaliency
import cv2
import sys
import numpy as np

for x in range(1,6):
    image = cv2.imread(str(sys.argv[x]))

    # cv2.imshow("test",image)
    width=len(image[0])
    height=len(image)
    newWidth = 400
    newHeight = int(height*(newWidth/width))
    salImage = getSaliency(image)
    salImage = cv2.resize(salImage, (newWidth, newHeight))
    image = cv2.resize(np.abs(image), (len(salImage[0]), len(salImage)))
    cv2.imshow("saliency",salImage)
    while True:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break