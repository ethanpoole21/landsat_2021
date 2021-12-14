# -*- coding: utf-8 -*-

"""
    @author: Ethan Poole 
"""
def lai(NIR,RED):
    import numpy as np

    nir = NIR.astype('float64')
    red = RED.astype('float64')

    lai = np.where((red)==0., 0, (nir)/(red))

    return lai