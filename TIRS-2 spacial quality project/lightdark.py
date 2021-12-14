# -*- coding: utf-8 -*-

"""
    @author: Ethan Poole 
"""

# imports
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import scipy
from numpy import genfromtxt


image = genfromtxt('roi.csv', delimiter=',')
image = image.astype(np.uint8)
roi = []

def click_event(event, x, y, flags, params):

    if event == cv2.EVENT_LBUTTONDOWN:
        print(x, ' ', y)

        roi.clear()
        roi.append([x,y])
 
if __name__=="__main__":
 
    img = image
 
    cv2.imshow('image', img)
    cv2.setMouseCallback('image', click_event)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
