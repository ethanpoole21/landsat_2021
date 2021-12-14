# -*- coding: utf-8 -*-

"""
    @author: Ethan Poole 
"""

import matplotlib.pyplot as plt
import numpy as np
import h5py
import cv2
import glob
import rasterio
import matplotlib.pylab as pylab
import scipy
import os
from scipy import stats
from ndvi import *
from evi2 import *
from savi import *
from ndmi import *
from ndwi import *
from ndsi import *
from lai import *
from colorimg import *


def read_hdf5(path):
    hf = h5py.File(path, 'r')
    adict = {}
    # Read attrs
    for attr in hf.attrs:
        adict[attr] = hf.attrs[attr]
    # Read datasets
    for key in hf.keys():
        adict[key] = hf[key][:]
    # Finish up
    hf.close()
    return adict

### band information ###

'''
10m bands
Band 3 Blue (490 μm)
Band 4 Green (560 μm)
Band 7 Red (665 μm)
Band 10 NIR_Broad (842 μm)
Band 17 SWIR 1 (1610 μm)
'''

# find directory for target images
target_dir1 = 'img1/'
target_dir2 = 'img2/'

flex_list = ['flex000_14bit-','flex005_14bit-','flex010_14bit-','flex020_14bit-','flex040_14bit-','flex080_14bit-',]

test_paths1 = sorted(glob.glob(target_dir1 + '*.img'))
headers1 = sorted(glob.glob(target_dir1 + '/*.hdr'))

test_paths2 = sorted(glob.glob(target_dir2 + '*.img'))
headers2 = sorted(glob.glob(target_dir2 + '/*.hdr'))


output1 = 'img1_output'
output2 = 'img2_output'

ndvi_output = []
evi2_output = []
ndmi_output = []
lai_output = []
rgb_output = []

ndvi_mean = []
evi2_mean = []
ndmi_mean = []
lai_mean = []

ndvi_stdev = []
evi2_stdev = []
ndmi_stdev = []
lai_stdev = []

ndvi_diff = []
evi2_diff = []
ndmi_diff = []
lai_diff = []

ndvi_percdiff = []
evi2_percdiff = []
ndmi_percdiff = []
lai_percdiff = []


n = 0 

print(test_paths1)

for path in test_paths1:

    data = rasterio.open(path)
    data = data.read()
  
    print(data.shape)

    blue = data[0,:,:]
    green = data[1,:,:]
    red = data[2,:,:]
    NIR = data[3,:,:]
    SWIR = data[4,:,:]

    rgb_out = makecolorimage(red,green,blue) # color image 
    rgb_out = np.ma.masked_where(rgb_out == np.nan, rgb_out)
    rgb_out = np.ma.masked_where(rgb_out == 0, rgb_out)

    ndvi_out = ndvi(NIR,red)  # Normalized difference vegetation index
    ndvi_out = np.ma.masked_where(ndvi_out == np.nan, ndvi_out)
    ndvi_out = np.ma.masked_where(ndvi_out == 0, ndvi_out)

    evi2_out = evi2(NIR,red)  # Enhanced Vegetation Index 2
    evi2_out = np.ma.masked_where(evi2_out == np.nan, evi2_out)
    evi2_out = np.ma.masked_where(evi2_out == 0, evi2_out)

    ndmi_out = ndmi(NIR,SWIR) # Normalized Difference Moisture Index
    ndmi_out = np.ma.masked_where(ndmi_out == np.nan, ndmi_out)
    ndmi_out = np.ma.masked_where(ndmi_out == 0, ndmi_out)

    lai_out = lai(NIR,red) # Normalized Difference Moisture Index
    lai_out = np.ma.masked_where(lai_out == np.nan, lai_out)
    lai_out = np.ma.masked_where(lai_out == 0, lai_out)

    ndvi_output.append(ndvi_out)
    evi2_output.append(evi2_out)
    ndmi_output.append(ndmi_out)
    lai_output.append(lai_out)
    rgb_output.append(rgb_out)

    if n == 0:
        ndvi_nocomp = ndvi_out
        evi2_nocomp = evi2_out
        ndmi_nocomp = ndmi_out
        lai_nocomp = lai_out

    else:
        ndvi_diff.append(ndvi_nocomp-ndvi_out)
        evi2_diff.append(evi2_nocomp-evi2_out)
        ndmi_diff.append(ndmi_nocomp-ndmi_out)
        lai_diff.append(lai_nocomp-ndmi_out)
    
    n += 1

roi = cv2.selectROI(rgb_output[0])

color = (255, 0, 0)
thickness = 2

start_point = int(roi[0]),int(roi[1])
end_point = int(roi[0]+roi[2]),int(roi[1]+roi[3])
roi_image = cv2.rectangle(rgb_output[0], start_point, end_point, color, thickness)

e = 0
for i in range(len(ndvi_output)):


    ndvi_cropped = ndvi_output[i][int(roi[1]):int(roi[1]+roi[3]), int(roi[0]):int(roi[0]+roi[2])]
    evi2_cropped = evi2_output[i][int(roi[1]):int(roi[1]+roi[3]), int(roi[0]):int(roi[0]+roi[2])]
    ndmi_cropped = ndmi_output[i][int(roi[1]):int(roi[1]+roi[3]), int(roi[0]):int(roi[0]+roi[2])]
    lai_cropped = lai_output[i][int(roi[1]):int(roi[1]+roi[3]), int(roi[0]):int(roi[0]+roi[2])]
    
    current_ndvi_mean = np.mean(ndvi_cropped)
    current_evi2_mean = np.mean(evi2_cropped)
    current_ndmi_mean = np.mean(ndmi_cropped)
    current_lai_mean = np.mean(lai_cropped)

    ndvi_mean.append(round(current_ndvi_mean,6))
    evi2_mean.append(round(current_evi2_mean,6))
    ndmi_mean.append(round(current_ndmi_mean,6))
    lai_mean.append(round(current_lai_mean,6))

    current_ndvi_stdev = np.std(ndvi_cropped)
    current_evi2_stdev = np.std(evi2_cropped)
    current_ndmi_stdev = np.std(ndmi_cropped)
    current_lai_stdev = np.std(lai_cropped)

    ndvi_stdev.append(round(current_ndvi_stdev,6))
    evi2_stdev.append(round(current_evi2_stdev,6))
    ndmi_stdev.append(round(current_ndmi_stdev,6))
    lai_stdev.append(round(current_lai_stdev,6))

    if e == 0:
        ndvi_roi_nocomp = current_ndvi_mean
        evi2_roi_nocomp = current_evi2_mean
        ndmi_roi_nocomp = current_ndmi_mean
        lai_roi_nocomp = current_lai_mean
        
        current_ndvi_percdiff = (np.mean(abs((ndvi_roi_nocomp-ndvi_cropped)/ndvi_roi_nocomp)*100))
        current_evi2_percdiff = (np.mean(abs((evi2_roi_nocomp-evi2_cropped)/evi2_roi_nocomp)*100))
        current_ndmi_percdiff = (np.mean(abs((ndmi_roi_nocomp-ndmi_cropped)/ndmi_roi_nocomp)*100))
        current_lai_percdiff = (np.mean(abs((lai_roi_nocomp-lai_cropped)/lai_roi_nocomp)*100))

        ndvi_percdiff.append(round(current_ndvi_percdiff,6))
        evi2_percdiff.append(round(current_evi2_percdiff,6))
        ndmi_percdiff.append(round(current_ndmi_percdiff,6))
        lai_percdiff.append(round(current_lai_percdiff,6))

    else:
        current_ndvi_percdiff = (np.mean(abs((ndvi_roi_nocomp-ndvi_cropped)/ndvi_roi_nocomp)*100))
        current_evi2_percdiff = (np.mean(abs((evi2_roi_nocomp-evi2_cropped)/evi2_roi_nocomp)*100))
        current_ndmi_percdiff = (np.mean(abs((ndmi_roi_nocomp-ndmi_cropped)/ndmi_roi_nocomp)*100))
        current_lai_percdiff = (np.mean(abs((lai_roi_nocomp-lai_cropped)/lai_roi_nocomp)*100))

        ndvi_percdiff.append(round(current_ndvi_percdiff,6))
        evi2_percdiff.append(round(current_evi2_percdiff,6))
        ndmi_percdiff.append(round(current_ndmi_percdiff,6))
        lai_percdiff.append(round(current_lai_percdiff,6))

    e+=1
params = {'legend.fontsize':5,
          'figure.figsize': (15, 5),
         'axes.labelsize': 10,
         'axes.titlesize':10,
         'xtick.labelsize':10,
         'ytick.labelsize':10}
pylab.rcParams.update(params)

fig1, (ax1,ax2,ax3,ax4,ax5) = plt.subplots(1,5)
plt.suptitle('sample images')
im1 = ax1.imshow(ndvi_nocomp)
fig1.colorbar(im1,ax=ax1)
ax1.set_title('ndvi')
im2 = ax2.imshow(evi2_nocomp)
fig1.colorbar(im2,ax=ax2)
ax2.set_title('evi2')
im3 = ax3.imshow(ndmi_nocomp)
fig1.colorbar(im3,ax=ax3)
ax3.set_title('ndmi')
im4 = ax4.imshow(lai_nocomp,vmin = np.percentile(lai_nocomp,2), vmax = np.percentile(lai_nocomp,98))
fig1.colorbar(im4,ax=ax4)
ax4.set_title('lai')
im5 = ax5.imshow(roi_image)
ax5.set_title('roi location')
plt.tight_layout()
plt.savefig(output1+'/output_indicie_graph.png',dpi = 600)


print('means:')
print(ndvi_mean)
print(evi2_mean)
print(ndmi_mean)
print(lai_mean)


print('stdev:')
print(ndvi_stdev)
print(evi2_stdev)
print(ndmi_stdev)
print(lai_stdev)

print('percdiff:')
print(ndvi_percdiff)
print(evi2_percdiff)
print(ndmi_percdiff)
print(lai_percdiff)

params = {'legend.fontsize':5,
          'figure.figsize': (5, 5),
         'axes.labelsize': 10,
         'axes.titlesize':10,
         'xtick.labelsize':10,
         'ytick.labelsize':10}
pylab.rcParams.update(params)

flex_list = [0,5,10,20,40,80]

fig2, [[ax1,ax2],[ax3,ax4]] = plt.subplots(2,2)
plt.suptitle('stdev plots')
im1 = ax1.scatter(flex_list,ndvi_stdev)
ax1.set_title('ndvi_stdev')
im2 = ax2.scatter(flex_list,evi2_stdev)
ax2.set_title('evi2_stdev')
im3 = ax3.scatter(flex_list,ndvi_stdev)
ax3.set_title('ndvi_stdev')
im4 = ax4.scatter(flex_list,lai_stdev)
ax4.set_title('lai_stdev')
plt.tight_layout()
plt.savefig(output1+'/output_stdev_plot.png',dpi = 600)

fig3, [[ax1,ax2],[ax3,ax4]] = plt.subplots(2,2)
plt.suptitle('mean plots')
im1 = ax1.scatter(flex_list,ndvi_mean)
ax1.set_title('ndvi_mean')
im2 = ax2.scatter(flex_list,evi2_mean)
ax2.set_title('evi2_mean')
im3 = ax3.scatter(flex_list,ndvi_mean)
ax3.set_title('ndvi_mean')
im4 = ax4.scatter(flex_list,lai_mean)
ax4.set_title('lai_mean')
plt.tight_layout()
plt.savefig(output1+'/output_mean_plot.png',dpi = 600)

fig4, [[ax1,ax2],[ax3,ax4]] = plt.subplots(2,2)
plt.suptitle('percend difference plots')
im1 = ax1.scatter(flex_list,ndvi_percdiff)
ax1.set_title('ndvi_percdiff')
im2 = ax2.scatter(flex_list,evi2_percdiff)
ax2.set_title('evi2_percdiff')
im3 = ax3.scatter(flex_list,ndvi_percdiff)
ax3.set_title('ndvi_percdiff')
im4 = ax4.scatter(flex_list,lai_percdiff)
ax4.set_title('lai_percdiff')
plt.tight_layout()
plt.savefig(output1+'/output_percdiff_plot.png',dpi = 600)

plt.rcParams.update(plt.rcParamsDefault)

title_text = 'ROI mean table'
footer_text = ' '
fig_background_color = 'skyblue'
fig_border = 'steelblue'

plt.figure(linewidth=2,
           edgecolor=fig_border,
           facecolor=fig_background_color,
           tight_layout={'pad':1},
           figsize=(4,3)
          )

cell_text = [(ndvi_mean),(evi2_mean),(ndmi_mean),(lai_mean)]
row_headers = ['ndvi_mean','evi2_mean','ndmi_mean','lai_mean']
column_headers = ['flex 0','flex 5','flex 10','flex 20','flex 40','flex 80']

rcolors = plt.cm.BuPu(np.full(len(row_headers), 0.1))
ccolors = plt.cm.BuPu(np.full(len(column_headers), 0.1))

the_table = plt.table(cellText=cell_text,
                      rowLabels=row_headers,
                      rowColours=rcolors,
                      rowLoc='center',
                      colColours=ccolors,
                      colLabels=column_headers,
                      loc='center')

the_table.scale(1, 1.1)

plt.suptitle(title_text)

ax = plt.gca()
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)

plt.box(on=None)

plt.figtext(0.95, 0.05, footer_text, horizontalalignment='center', size='medium')
plt.draw()
plt.tight_layout()

fig = plt.gcf()
plt.savefig(output1+'/mean_table.png',
            bbox='tight',
            edgecolor=fig.get_edgecolor(),
            facecolor=fig.get_facecolor(),
            dpi=600
            )

title_text = 'ROI stdev table'
footer_text = ' '
fig_background_color = 'skyblue'
fig_border = 'steelblue'

plt.figure(linewidth=2,
           edgecolor=fig_border,
           facecolor=fig_background_color,
           tight_layout={'pad':1},
           figsize=(4,3)
          )

cell_text = [(ndvi_stdev),(evi2_stdev),(ndmi_stdev),(lai_stdev)]
row_headers = ['ndvi_stdev','evi2_stdev','ndmi_stdev','lai_stdev']
column_headers = ['flex 0','flex 5','flex 10','flex 20','flex 40','flex 80']

rcolors = plt.cm.BuPu(np.full(len(row_headers), 0.1))
ccolors = plt.cm.BuPu(np.full(len(column_headers), 0.1))

the_table = plt.table(cellText=cell_text,
                      rowLabels=row_headers,
                      rowColours=rcolors,
                      rowLoc='center',
                      colColours=ccolors,
                      colLabels=column_headers,
                      loc='center')

the_table.scale(1, 1.1)

plt.suptitle(title_text)

ax = plt.gca()
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)

plt.box(on=None)

plt.figtext(0.95, 0.05, footer_text, horizontalalignment='center', size='medium')
plt.draw()
plt.tight_layout()

fig = plt.gcf()
plt.savefig(output1+'/stdev_table.png',
            bbox='tight',
            edgecolor=fig.get_edgecolor(),
            facecolor=fig.get_facecolor(),
            dpi=600
            )

title_text = 'ROI percdiff table'
footer_text = ' '
fig_background_color = 'skyblue'
fig_border = 'steelblue'

plt.figure(linewidth=2,
           edgecolor=fig_border,
           facecolor=fig_background_color,
           tight_layout={'pad':1},
           #figsize=(4,3)
          )

cell_text = [(ndvi_percdiff),(evi2_percdiff),(ndmi_percdiff),(lai_percdiff)]
row_headers = ['ndvi_percdiff','evi2_percdiff','ndmi_percdiff','lai_percdiff']
column_headers = ['flex 0','flex 5','flex 10','flex 20','flex 40','flex 80']

rcolors = plt.cm.BuPu(np.full(len(row_headers), 0.1))
ccolors = plt.cm.BuPu(np.full(len(column_headers), 0.1))

the_table = plt.table(cellText=cell_text,
                      rowLabels=row_headers,
                      rowColours=rcolors,
                      rowLoc='center',
                      colColours=ccolors,
                      colLabels=column_headers,
                      loc='center')

the_table.scale(1, 1.5)

plt.suptitle(title_text)

ax = plt.gca()
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)

plt.box(on=None)

plt.figtext(.25, 0, footer_text, horizontalalignment='center', size='large')
plt.draw()
plt.tight_layout()

fig = plt.gcf()
plt.savefig(output1+'/percdiff_table.png',
            bbox='tight',
            edgecolor=fig.get_edgecolor(),
            facecolor=fig.get_facecolor(),
            dpi=600
            )