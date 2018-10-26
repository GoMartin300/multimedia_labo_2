import sys
import cv2
import numpy as np

cap = cv2.VideoCapture(str(sys.argv[1]))

ret, frame = cap.read()
frame = cv2.resize(frame, (0,0), fx=0.3, fy=0.3)
height = frame.shape[0]
width = frame.shape[1]

while(cap.isOpened()):

    ret, frame = cap.read()

    frame = cv2.resize(frame, (0,0), fx=0.3, fy=0.3)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    cv2.imshow('origineel', frame)   
   
    for r in range(0, height-1):
    	for c in range(0, width-1):
    		#pixeltransformations
    		
    		frame.itemset((r,c,0),255 - frame.item(r,c,0))
    		frame.itemset((r,c,1),255 - frame.item(r,c,1))
    		frame.itemset((r,c,2),255 - frame.item(r,c,2))

    cv2.imshow('resultaat', frame)   
   	    		    		    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cap.release()

while(True):
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
