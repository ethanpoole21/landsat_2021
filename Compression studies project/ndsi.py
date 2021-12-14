# -*- coding: utf-8 -*-

"""
    @author: Ethan Poole 
"""
def ndsi(GREEN,SWIR):
    import numpy as np

    np.seterr(divide='ignore', invalid='ignore')

    green = GREEN.astype('float64')
    swir = SWIR.astype('float64')
    ndsi=np.where((green+swir)==0., 0, (green-swir)/(green+swir))

    return ndsi
