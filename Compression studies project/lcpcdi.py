# -*- coding: utf-8 -*-

"""
    @author: Ethan Poole 
"""

def lcpcdi(SWIR_2b,SWIR_2c):
    import numpy as np

    #Ligno-Cellulose Peak Centered Difference Index
    
    b = SWIR_2b.astype('float64')
    c = SWIR_2c.astype('float64')

    lcpcdi = (2*(b))- (b+ c)

    return lcpcdi