import os
import sys
import argparse
import cv2
import glob
import numpy as np
from PIL import Image
import scipy
from scipy import stats
import csv

np.set_printoptions(threshold=np.inf)

parser = argparse.ArgumentParser(
    prog="coeff",
    usage="cal_coeff_generator.py [-h] [-t] target_dir [flags ...]",
    description="Generate callibration coefficients for A655sc FLIR camera"
)
parser.add_argument(
    "-t",
    "--target_dir",
    required=True,
    default="fullcal2421",
    help="Location of directory with target callibration images."
)
parser.add_argument(
    "-o",
    "--output",
    default="coefficients.csv",
    help="Path to export final coefficients as CSV file."
)
parser.add_argument(
    "-fwhm",
    "--full_width_half_maximum",
    default= 1,
    help="Path to export final coefficients as CSV file."
)
parser.add_argument(
    "-c",
    "--center",
    default= 9.5,
    help="Path to export final coefficients as CSV file."
)


args = parser.parse_args()

# blackbody curve
def blackbody(T, w):
    """ Returns the emission radiance of a blackbody
        param[in]   T   Temperature [k] 
        param[in]   w   Wavelength [nm] 
        param[out]  L   Blackbody radiance [W/m^-2/sr^-1/micron]    
    """
    h = 6.62607015e-34 # Planck Constant
    k = 1.380649e-23 # Boltzmann Constant
    c = 2.99792458e8 # Speed of light

    L = (2*h*c*c)/((w**5)*(np.exp((h*c)/(w*k*T))-1)) * 1e-6

    return L

# generate gaussian curve for response
def get_gaus_band(wvl,band_center,fwhm):
    """ Returns normal distribution for given parameters
        param[in]   wvl             list of wavelengths
        param[in]   band_center     center wavelength
        param[in]   fwhm            full width half maximum
        param[out]  y               normalised probability density function   
    """
    y = scipy.stats.norm.pdf(wvl,band_center,fwhm/2.3548)
    y = y/max(y)
    return y

dir_path = args.target_dir

num_folders = (len(os.listdir(dir_path)))

for i in range(num_folders):
    globals()['path_%s' % i] = dir_path + '/' +str(10+i*5)
    globals()['files_%s' % i] = glob.glob(os.path.join(globals()['path_%s' % i], "*.tif"))

image_list = os.listdir(path_1)

img = np.asarray(Image.open(path_1 + '/' +image_list[1]))

dims =np.asarray(img).shape

N=len(img)

h = dims[0]
w = dims[1]

for i in range(num_folders):
    globals()['arr_%s' % i] = np.zeros((h,w),np.float)

for i in range(num_folders):
    for im in (globals()['files_%s' % i]):
        imarr = np.array(cv2.imread(im,-1),dtype=np.float32)
        (globals()['arr_%s' % i]) = (globals()['arr_%s' % i]) + imarr
    (globals()['array_%s' % i]) = (globals()['arr_%s' % i])/N
    (globals()['average_%s' % i]) = np.asarray((globals()['array_%s' % i]))
    (globals()['flat_average_%s' % i]) = (globals()['average_%s' % i]).flat

for i in range(num_folders):
    (globals()['bb_%s' % i]) = []
    (globals()['temp%s' % i]) = 273.15 + (10+i*5)
    for j in np.arange(7e-6, 14e-6, .01e-6):
        (globals()['bb_%s' % i]).append(blackbody((globals()['temp%s' % i]),j))

wavelengths = []
center = args.center
fwhm = args.full_width_half_maximum
for i in np.arange(7, 14, .01):
    wavelengths.append(i)
resp = get_gaus_band(wavelengths,center,fwhm)

for i in range(num_folders):
        (globals()['response_%s' % i]) = (globals()['bb_%s' % i]) * resp

        (globals()['integrate_%s' % i]) = np.trapz((globals()['response_%s' % i]),wavelengths)
        (globals()['integrate_resp_%s' % i]) = np.trapz(resp,wavelengths)
        (globals()['final_%s' % i]) = (globals()['integrate_%s' % i]) / (globals()['integrate_resp_%s' % i])

bb_values = []
for i in range(num_folders):
        bb_values.append((globals()['final_%s' % i]))


with open(args.output, 'w', newline='') as csvfile:
    file = csv.writer(csvfile, delimiter=',',quotechar='|', quoting=csv.QUOTE_MINIMAL)
    file.writerow(bb_values)
