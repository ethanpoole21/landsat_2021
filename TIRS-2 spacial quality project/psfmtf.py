# -*- coding: utf-8 -*-

"""
    @author: Ethan Poole 
"""

# imports
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import scipy
import math
import scipy.optimize as optimization
from PIL import Image 
import PIL
from numpy import diff
from scipy.signal import savgol_filter
import lmfit
from lmfit import Model

dark_roi = [[10,9]]
light_roi = [[48,48]]

def fermi_function(b,d,x,s,g,e):
    ans = d + (b-d) / (1 + np.exp(-s * (x - e) )) + (g*x)
    return ans
x = np.arange(0,50,1)
def residual(params):
    s = params['s']
    e = params['e']
    g = params['g']
    y = fermi_function(b,d,x,s,g,e)
    res = (savgol_filter(row,11, 3) - y)**2  
    return res

def peak(x, c):
    return np.exp(-np.power(x - c, 2) / 16.0)

def lin_interp(x, y, i, half):
    return x[i] + (x[i+1] - x[i]) * ((half - y[i]) / (y[i+1] - y[i]))

def half_max_x(x, y):
    half = max(y)/2.0
    signs = np.sign(np.add(y, -half))
    zero_crossings = (signs[0:-2] != signs[1:-1])
    zero_crossings_i = np.where(zero_crossings)[0]
    return [lin_interp(x, y, zero_crossings_i[0], half),
            lin_interp(x, y, zero_crossings_i[1], half)]

# upload image
path = 'psf_mtf/on/LC08_L1TP_157045_20211108_20211117_02_T1 copy/'
im = 'LC08_L1TP_157045_20211108_20211117_02_T1_B10.TIF'
imname = 'L8 11/'
output = 'l811_08_157045/'
image = Image.open(path+im)
imarray = np.array(image)
temp_im = imarray #(imarray*0.00341802+149)-273.15 converts to temp
low_values_flags = temp_im < 0  # Where values are low
temp_im[low_values_flags] = 0 
temp_im = temp_im.astype(np.float)

plt.figure()
plt.imshow(temp_im,cmap='gray')
plt.colorbar()
plt.title('imname')
plt.savefig(output+'image.png', dpi=300)

kernel = np.genfromtxt('roi.csv', delimiter=',')

plt.figure()
plt.imshow(kernel,cmap='gray')
plt.title(imname+' kernel')
plt.colorbar()
plt.savefig('LC08_L1TP_157045_20211108_20211117_02_T1_B10_kernel.png', dpi=300)
np.savetxt(output+'roi.csv', kernel, delimiter=',')


for row in kernel:
    plt.plot(row)
plt.figure()
plt.title(imname+' profiles')
plt.savefig(output+'profiles.png', dpi=300)


d_cent = dark_roi[0]
dx_cent = d_cent[0]
dy_cent = d_cent[1]

b_cent = light_roi[0]
bx_cent = b_cent[0]
by_cent = b_cent[1]

db_pad = 3

d_kernel = kernel[dy_cent - db_pad:dy_cent + db_pad, dx_cent - db_pad:dx_cent + db_pad]
b_kernel = kernel[by_cent - db_pad:by_cent + db_pad, bx_cent - db_pad:bx_cent + db_pad]

d = np.mean(d_kernel)
b = np.mean(b_kernel)

for row in kernel:
    x0 = np.array([0.0, 0.0, 0.0])
    sigma = np.array(np.ones(len(row)))
    xdata= np.array(list(range(0, len(row))))
    xdata = xdata.astype(np.float64)
    row = row.astype(np.float64)

    params = lmfit.Parameters()
    params.add('s', min=0.1, max=2.0)
    params.add('e', min=-75.0, max=75.0)
    params.add('g', min=0.0, max=14.0)

    mini = lmfit.Minimizer(residual, params, nan_policy='propagate')
    out1 = mini.minimize(method='Nelder')
    out2 = mini.minimize(method='leastsq', params=out1.params)
    p = (out2.params)
    
    e_val = p['e']._val
    curr = e_val
    xdata = xdata- curr
    plt.plot(xdata,row)

plt.title(imname+' profiles aligned')
plt.savefig(output+'aligned_profiles.png', dpi=300)

fwhm = []
snr = []

for row in kernel:
    x0 = np.array([0.0, 0.0, 0.0])
    sigma = np.array(np.ones(len(row)))
    xdata= np.array(list(range(0, len(row))))
    xdata = xdata.astype(np.float64)
    row = row.astype(np.float64)

    params = lmfit.Parameters()
    params.add('s', min=0.1, max=2.0)
    params.add('e', min=-75.0, max=75.0)
    params.add('g', min=0.0, max=14.0)

    mini = lmfit.Minimizer(residual, params, nan_policy='propagate')
    out1 = mini.minimize(method='Nelder')
    out2 = mini.minimize(method='leastsq', params=out1.params)
    p = (out2.params)
    
    s_val = p['s']._val
    e_val = p['e']._val
    g_val = p['g']._val

    curr = e_val

    xdata = xdata- curr
    
    norm_esf = ((savgol_filter(row,11, 3) - d - g_val*xdata) / (b-d))
    y_norm = ((savgol_filter(row,11,3) - d - g_val*xdata) / (b-d))
    diff_row = np.gradient(y_norm)
    norm_diff_row = diff_row/np.max(diff_row)

    hmx = half_max_x(xdata,diff_row)
    curr_fwhm = hmx[1] - hmx[0]
    fwhm.append(curr_fwhm)

    axis = x - e_val
    bright = np.argwhere(axis > int(curr_fwhm+1)).flatten()
    dark = np.argwhere(axis < -int(curr_fwhm+1)).flatten()

    if (len(dark) > 0) and (len(bright) > 0):
    
        mean_bright = np.mean(savgol_filter(row,11, 3)[bright])
        mean_dark = np.mean(savgol_filter(row,11, 3)[dark])
    
        std_bright = np.std(savgol_filter(row,11, 3)[bright])
        std_dark = np.std(savgol_filter(row,11, 3)[dark])
    
        v_difference = mean_bright - mean_dark
        curr_snr= v_difference / ((std_bright + std_dark)/2.)
        
    else:
        curr_snr = np.nan

    snr.append(curr_snr)

    plt.plot(xdata,norm_diff_row)

plt.title(imname+'normalised point spread function')
plt.savefig(output+'psf.png', dpi=300)


plt.show()
print('full width half max = ' , np.mean(fwhm))
spectral_resolution = np.mean(fwhm)*30
print('spectral resolution = ' , spectral_resolution)
print('signal to noise ratio = ' , np.mean(snr))