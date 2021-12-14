# -*- coding: utf-8 -*-

"""
    @author: Ethan Poole 
"""
def savi(NIR,RED,L):
    import numpy as np

    # L is The amount of green vegetation cover (0-1)
    
    nir = NIR.astype('float64')
    red = RED.astype('float64')

    savi = ((nir - red)/(nir + red + L)) * (1 + L)

    return savi