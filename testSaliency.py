from spectralresidualsaliency import getSaliency
import cv2
import sys
import numpy as np

for x in range(1,5):
    image = cv2.imread(str(sys.argv[x]))

    # cv2.imshow("test",image)
    width=image.shape[0]
    height=image.shape[1]
    newWidth = 400
    newHeight = height*(newWidth/width)
    image = cv2.resize(image, (newWidth, int(newHeight)))
    cv2.imshow('x', np.concatenate((image, getSaliency(image)), axis=1))
    while True:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

