# -*- coding: utf-8 -*-

"""
    @author: Ethan Poole 
"""
def ndvi(NIR,RED):
    import numpy as np

    np.seterr(divide='ignore', invalid='ignore')

    nir = NIR.astype('float64')
    red = RED.astype('float64')
    ndvi=np.where((nir+red)==0., 0, (nir-red)/(nir+red))

    return ndvi
