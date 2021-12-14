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
import math
import scipy.optimize as optimization
from PIL import Image 
from numpy import diff
from scipy.signal import savgol_filter
import lmfit
from lmfit import Model

# find image
path = 'psf_mtf/on/LC08_L1TP_157045_20211108_20211117_02_T1 copy/'
im = 'LC08_L1TP_157045_20211108_20211117_02_T1_B10.TIF'
image = Image.open(path+im)
imarray = np.array(image)
temp_im = imarray #(imarray*0.00341802+149)-273.15 to turn image into temp if you want
low_values_flags = temp_im < 0  # Where values are low
temp_im[low_values_flags] = 0 
temp_im = temp_im.astype(np.uint16)

f = plt.figure()
plt.imshow(temp_im)
plt.colorbar()
plt.show()

axischeck = input("Enter 0 for on track or 1 for off track:")
check = int(axischeck)

roi = []
dark = []
light = []

# functions 
def click_event(event, x, y, flags, params):

    if event == cv2.EVENT_LBUTTONDOWN:
        print(x, ' ', y)
        roi.clear()
        roi.append([x,y])

def roi_click_event(event, x, y, flags, params):

    if event == cv2.EVENT_LBUTTONDOWN:
        print(x, ' ', y)
        dark.clear()
        dark.append([x,y])
    elif event == cv2.EVENT_RBUTTONDOWN:
        print(x, ' ', y)
        light.clear()
        light.append([x,y])
 
if __name__=="__main__":
    
    
    if check == 1:
        temp_im = np.rot90(temp_im)
        print('com')

    img = temp_im


 
    cv2.imshow('image', img)
    cv2.setMouseCallback('image', click_event)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

print(roi)

cent = roi[0]
x_cent = cent[0]
y_cent = cent[1]

pad = 25

kernel = temp_im[y_cent - pad:y_cent + pad, x_cent - pad:x_cent + pad]

f = plt.figure()
plt.imshow(kernel)
plt.colorbar()
plt.show()

np.savetxt('roi.csv', kernel, delimiter=',')

# main
if __name__=="__main__":
    
    img = kernel
    img = img.astype(np.uint16)
 
    cv2.imshow('image', img)
    cv2.setMouseCallback('image', roi_click_event)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

d_cent = dark[0]
dx_cent = d_cent[0]
dy_cent = d_cent[1]

b_cent = light[0]
bx_cent = b_cent[0]
by_cent = b_cent[1]

print(d_cent,b_cent)