import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import cv2
import glob
import rasterio
from envi_header import find_hdr_file,read_hdr_file,write_envi_header
import spectral.io.envi as envi

# import test images
target_dir = 'test'
test_paths = glob.glob(target_dir + '/*.tif')
multiple_length = len(test_paths)

shape_img = rasterio.open(test_paths[1])
shape_img = shape_img.read(1)
dims =np.asarray(shape_img).shape


def bb(wave,temp):
    #returns radiance
    h = 6.626 * 10**(-34)
    c = 2.998 * 10**8
    k = 1.381 * 10**(-23)
    wave = wave * 1e-6 #um to m
    emission = 2*h*c**(2) / (wave**(5) * (np.exp((h*c) / (wave * temp * k))-1)) * 10**(-6)
    return emission #W/m^2 sr um
def create_RSR(center, FWHM):
    #wave = np.arange(center-2,center+2,0.01)
    wave = np.arange(7.5,10.5,0.01)
    rsr = norm.pdf(wave,center,FWHM/2.3548)
    rsr = rsr/max(rsr)
    #plt.plot(wave,rsr)
    return rsr
#make LUT
wave = np.arange(7.5,10.5,.01)
rsr = create_RSR(9,1)
LUT = []
lut_temps  = np.arange(250,375,0.1)
for i in range(lut_temps.shape[0]):
    band_eff = np.trapz(list(rsr * bb(wave,lut_temps[i])),x=list(wave),axis=0)/np.trapz(list(rsr),x=list(wave),axis=0)
    # print(band_eff)
    LUT.append(((lut_temps[i]-273.15),band_eff))
LUT = np.asarray(LUT)

#apply lut
rad_LUTim = np.ones((dims[0],dims[1],LUT[:,0].shape[0]))*LUT[:,1]

g = 1
print(len(test_paths))
for i in range (len(test_paths)):
    print(i)
    image = (cv2.imread(test_paths[i],-1)).astype(np.float32)

    test = (abs(rad_LUTim - image[...,np.newaxis])).argmin(axis =2)

    temp = LUT[:,0][test.flatten()]
    current_img = (temp.reshape((dims[0],dims[1])))

    cv2.imwrite('lut_images/image'+str(g) + '.tif', current_img)

    fig, ax = plt.subplots()
    im = ax.imshow(current_img, cmap=plt.get_cmap('gray'),vmin = np.percentile(current_img,2), vmax = np.percentile(current_img,98))
    print(np.percentile(current_img,2))
    print(np.percentile(current_img,98))
    fig.colorbar(im)
    plt.savefig('lut_plots/image'+str(g) + '.png')

    g +=1
