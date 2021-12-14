# -*- coding: utf-8 -*-

"""
    @author: Ethan Poole 
"""

def evi2(NIR,RED):
    import numpy as np
    
    nir = NIR.astype('float64')
    red = RED.astype('float64')

    evi2 = (2.4*((nir - red) / (nir + red + 1)))

    return evi2