# -*- coding: utf-8 -*-

"""
    @author: Ethan Poole 
"""
def ndmi(NIR,SWIR):
    import numpy as np

    nir = NIR.astype('float64')
    swir = SWIR.astype('float64')

    ndmi = np.where((nir+swir)==0., 0, (nir - swir)/(nir + swir))

    return ndmi