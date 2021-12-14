import os
import sys
import argparse
import cv2
import numpy as np
from PIL import Image
import csv
import glob
import rasterio
import matplotlib.pyplot as plt
from envi_header import find_hdr_file,read_hdr_file,write_envi_header
import spectral.io.envi as envi

# inverse blackbody
def invblackbody(l):
    """ Returns inverted Planck Function to find objects temperature
        param[out]  T   Temperature [k] 
        param[in]   w   Wavelength [nm] 
        param[in]   L   Blackbody radiance [W/m^-2/sr^-1/micron]    
    """
    w = 10

    W = w * 1e-6
    L = l * 1e6
    h = 6.62607015e-34 # Planck Constant
    k = 1.380649e-23 # Boltzmann Constant
    c = 2.99792458e8 # Speed of light

    T =(h*c/(k*W))/(np.log((2*h*c*c/(L*(W**5)))+1))
    return T

target_dir = 'rad_images'

test_paths = glob.glob(target_dir + '/*.img')
headers = glob.glob(target_dir + '/*.hdr')

multiple_length = len(test_paths)
print(multiple_length)

g = 1

for e in range (multiple_length):    

    image_file = test_paths[e]
    image_hdr_file = headers[e]

    in_header = find_hdr_file(image_file)
    header_data = read_hdr_file(in_header)
 
    current_img = envi.open(image_hdr_file,image_file)
    current_img = current_img.open_memmap(writable=True)

    fig, ax = plt.subplots()
    im = ax.imshow(current_img, cmap=plt.get_cmap('gray'),vmin = np.percentile(current_img,2), vmax = np.percentile(current_img,98))
    fig.colorbar(im)
    plt.show()
    
    #
    dims =np.asarray(current_img).shape

    r = dims[0]
    c = dims[1]

    q = r*c
    flat_current_img = current_img.flat

    current_im_temp = []
    for x in range(q):
        current_im_temp.append((invblackbody(flat_current_img[x]))-273.15)
    
    fin_temps = np.asarray(current_im_temp).reshape(r,c)

    envi.save_image('invplanc_images/lut_image'+str(g)+'.hdr',fin_temps, force=True, dtype=float)

    output_header_dict = read_hdr_file(find_hdr_file(headers[1]))
    output_header_dict['description'] = 'put this in the command line at some point'
    output_header_dict['lines'] = str(fin_temps.shape[0])
    output_header_dict['samples'] = str(fin_temps.shape[1])
    output_header_dict['interleave'] = 'bip'

    write_envi_header('invplanc_images/lut_image'+str(g)+'.hdr',output_header_dict)

    fig, ax = plt.subplots()
    im = ax.imshow(fin_temps, cmap=plt.get_cmap('gray'),vmin = np.percentile(fin_temps,2), vmax = np.percentile(fin_temps,98))
    fig.colorbar(im)
    plt.savefig('invplanc_plots/image'+ str(g) + '.png')
    plt.show()

    g +=1

