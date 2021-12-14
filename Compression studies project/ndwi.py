# -*- coding: utf-8 -*-

"""
    @author: Ethan Poole 
"""
def ndwi(GREEN,NIR):
    import numpy as np

    nir = NIR.astype('float64')
    green = GREEN.astype('float64')

    ndwi = (green - nir) / (green + nir)

    return ndwi