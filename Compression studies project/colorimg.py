# -*- coding: utf-8 -*-

"""
    @author: Ethan Poole 
"""

def makecolorimage(RED,GREEN,BLUE):
    import cv2
    import numpy as np
    from skimage import exposure

    red = (RED.astype('float64') - np.min(RED))/(np.max(RED) - np.min(RED))
    green = (GREEN.astype('float64') - np.min(GREEN))/(np.max(GREEN) - np.min(GREEN))
    blue = (BLUE.astype('float64')- np.min(BLUE))/(np.max(BLUE) - np.min(BLUE))

    merged = cv2.merge((blue, green, red))

    # Contrast stretching
    p2 = np.percentile(merged, 2)
    p98 = np.percentile(merged, 98)
    merged = exposure.rescale_intensity(merged, in_range=(p2, p98))

    return merged