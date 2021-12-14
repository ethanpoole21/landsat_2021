# -*- coding: utf-8 -*-

"""
    @author: Ethan Poole 
"""

def sindri(SWIR_2b,SWIR_2c):
    import numpy as np

    #Shortwave Infrared Normalized Difference Residue Index
    
    b = SWIR_2b.astype('float64')
    c = SWIR_2c.astype('float64')

    sindri=np.where((b + c)==0., 0,(b - c) / (b + c))

    return sindri