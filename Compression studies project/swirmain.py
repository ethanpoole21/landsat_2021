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
from sindri import *
from lcpcdi import *
from colorimg import *



params = {'legend.fontsize':5,
          'figure.figsize': (15, 5),
         'axes.labelsize': 10,
         'axes.titlesize':10,
         'xtick.labelsize':10,
         'ytick.labelsize':10}
pylab.rcParams.update(params)

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

def mse(imageA, imageB):
	# the 'Mean Squared Error' between the two images is the
	# sum of the squared difference between the two images;
	err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
	err /= float(imageA.shape[0] * imageA.shape[1])
	
	# return the MSE, the lower the error, the more "similar"
	# the two images are
	return err


def signaltonoise(a, axis, ddof):
    a = np.asanyarray(a)
    m = a.mean(axis)
    sd = a.std(axis = axis, ddof = ddof)
    return np.where(sd == 0, 0, m / sd)

### band information ###
'''
10m bands
Band 3 Blue (490 μm)
Band 4 Green (560 μm)
Band 7 Red (665 μm)
Band 10 NIR_Broad (842 μm)
Band 17 SWIR 1 (1610 μm)

20m bands
Band 2 Coastal Aerosol (433 μm)
Band 5 Orange (620 μm)
Band 6 Red 1 (650 μm)
Band 8 Red Edge 1 (705 μm)B
and 9 Red Edge 2 (740 μm)
Band 11 NIR 1 (865 μm)
Band 13 Liquid Water (985 μm)
Band 14 Snow/Ice 1 (1035 μm)
Band 15 Snow/Ice 2 (1090 μm)
Band 18 SWIR 2a (2100 μm)
Band 19 SWIR 2b (2210 μm)
Band 20 SWIR 2c (2260 μm)

60M bands
Band 1 Violet (410 μm)
Band 12 Water Vapor (945 μm)
Band 16 Cirrus (1375 μm)
'''

# find directory for target images
target_dir = 'img1/'
target_dir2 = 'img220/'

flex_list = ['flex000_14bit-','flex005_14bit-','flex010_14bit-','flex020_14bit-','flex040_14bit-','flex080_14bit-',]

test_paths = sorted(glob.glob(target_dir + '*.img'))
headers1 = sorted(glob.glob(target_dir + '/*.hdr'))

test_paths2 = sorted(glob.glob(target_dir2 + '*.img'))
headers2 = sorted(glob.glob(target_dir2 + '/*.hdr'))

output = 'swir_outputs2b'

sindri_output = []
lcpcdi_output = []

sindri_mean = []
lcpcdi_mean = []

sindri_stdev = []
lcpcdi_stdev = []

sindri_diff = []
lcpcdi_diff = []

sindri_percdiff = []
lcpcdi_percdiff = []

n = 0 

print(test_paths)

for path in test_paths2:

    path20 = path
    path20 = rasterio.open(path20)
    path20 = path20.read()


    print(path20.shape)

    Coastal_Aerosol = path20[0,:,:]
    Orange = path20[1,:,:]
    Red_1 = path20[2,:,:]
    Red_Edge_1 = path20[3,:,:]
    Red_Edge_2 = path20[4,:,:]
    NIR_1= path20[5,:,:]
    Liquid_Water = path20[6,:,:]
    SnowIce_1 = path20[7,:,:]
    SnowIce_1 = path20[8,:,:]
    SWIR_2a = path20[9,:,:]
    SWIR_2b = path20[10,:,:]
    SWIR_2c = path20[11,:,:]

    sindri_out = sindri(SWIR_2b,SWIR_2c)  # Normalized difference vegetation index
    sindri_out = np.ma.masked_where(sindri_out == np.nan, sindri_out)
    #sindri_out = np.ma.masked_where(sindri_out == 0, sindri_out)

    lcpcdi_out = lcpcdi(SWIR_2b,SWIR_2c)  # Enhanced Vegetation Index 2
    lcpcdi_out = np.ma.masked_where(lcpcdi_out == np.nan, lcpcdi_out)
    #lcpcdi_out = np.ma.masked_where(lcpcdi_out == 0, lcpcdi_out)

    sindri_output.append(sindri_out)
    lcpcdi_output.append(lcpcdi_out)

    if n == 0:
        sindri_nocomp = sindri_out
        lcpcdi_nocomp = lcpcdi_out

    else:
        sindri_diff.append(sindri_nocomp-sindri_out)
        lcpcdi_diff.append(lcpcdi_nocomp-lcpcdi_out)

    n += 1


roi = cv2.selectROI(sindri_output[1])

color = (255, 0, 0)
thickness = 2

start_point = int(roi[0]),int(roi[1])
end_point = int(roi[0]+roi[2]),int(roi[1]+roi[3])
roi_image = cv2.rectangle(sindri_out, start_point, end_point, color, thickness)

e = 0
for i in range(len(sindri_output)):

    sindri_cropped = sindri_output[i][int(roi[1]):int(roi[1]+roi[3]), int(roi[0]):int(roi[0]+roi[2])]
    lcpcdi_cropped = lcpcdi_output[i][int(roi[1]):int(roi[1]+roi[3]), int(roi[0]):int(roi[0]+roi[2])]

    
    current_sindri_mean = np.mean(sindri_cropped)
    current_lcpcdi_mean = np.mean(lcpcdi_cropped)

    sindri_mean.append(round(current_sindri_mean,6))
    lcpcdi_mean.append(round(current_lcpcdi_mean,6))


    current_sindri_stdev = np.std(sindri_cropped)
    current_lcpcdi_stdev = np.std(lcpcdi_cropped)


    sindri_stdev.append(round(current_sindri_stdev,6))
    lcpcdi_stdev.append(round(current_lcpcdi_stdev,6))

    if e == 0:
        sindri_roi_nocomp = current_sindri_mean
        lcpcdi_roi_nocomp = current_lcpcdi_mean

        
        current_sindri_percdiff = (np.mean(abs((sindri_roi_nocomp-sindri_cropped)/sindri_roi_nocomp)*100))
        current_lcpcdi_percdiff = (np.mean(abs((lcpcdi_roi_nocomp-lcpcdi_cropped)/lcpcdi_roi_nocomp)*100))

        sindri_percdiff.append(round(current_sindri_percdiff,6))
        lcpcdi_percdiff.append(round(current_lcpcdi_percdiff,6))

    else:
        current_sindri_percdiff = (np.mean(abs((sindri_roi_nocomp-sindri_cropped)/sindri_roi_nocomp)*100))
        current_lcpcdi_percdiff = (np.mean(abs((lcpcdi_roi_nocomp-lcpcdi_cropped)/lcpcdi_roi_nocomp)*100))

        sindri_percdiff.append(round(current_sindri_percdiff,6))
        lcpcdi_percdiff.append(round(current_lcpcdi_percdiff,6))

    e+=1

params = {'legend.fontsize':5,
          'figure.figsize': (10, 5),
         'axes.labelsize': 10,
         'axes.titlesize':10,
         'xtick.labelsize':10,
         'ytick.labelsize':10}
pylab.rcParams.update(params)

fig1, (ax1,ax2,ax3) = plt.subplots(1,3)
plt.suptitle('sample images')
im1 = ax1.imshow(sindri_nocomp)
fig1.colorbar(im1,ax=ax1)
ax1.set_title('sindri')
im2 = ax2.imshow(lcpcdi_nocomp)
fig1.colorbar(im2,ax=ax2)
ax2.set_title('lcpcdi')
im3 = ax3.imshow(roi_image)
ax3.set_title('roi location')
plt.tight_layout()
plt.savefig(output+'/output_indicie_graph.png',dpi = 600)


print('means:')
print(sindri_mean)
print(lcpcdi_mean)


print('stdev:')
print(sindri_stdev)
print(lcpcdi_stdev)

print('percdiff:')
print(sindri_percdiff)
print(lcpcdi_percdiff)


params = {'legend.fontsize':5,
          'figure.figsize': (5, 5),
         'axes.labelsize': 10,
         'axes.titlesize':10,
         'xtick.labelsize':10,
         'ytick.labelsize':10}
pylab.rcParams.update(params)

flex_list = [0,5,10,20,40,80]

fig2, (ax1,ax2) = plt.subplots(1,2)
plt.suptitle('stdev plots')
im1 = ax1.scatter(flex_list,sindri_stdev)
ax1.set_title('sindri_stdev')
im2 = ax2.scatter(flex_list,lcpcdi_stdev)
ax2.set_title('lcpcdi_stdev')
plt.tight_layout()
plt.savefig(output+'/output_stdev_plot.png',dpi = 600)

fig3, (ax1,ax2) = plt.subplots(1,2)
plt.suptitle('mean plots')
im1 = ax1.scatter(flex_list,sindri_mean)
ax1.set_title('sindri_mean')
im2 = ax2.scatter(flex_list,lcpcdi_mean)
ax2.set_title('lcpcdi_mean')
plt.tight_layout()
plt.savefig(output+'/output_mean_plot.png',dpi = 600)

fig4, (ax1,ax2) = plt.subplots(1,2)
plt.suptitle('percend difference plots')
im1 = ax1.scatter(flex_list,sindri_percdiff)
ax1.set_title('sindri_percdiff')
im2 = ax2.scatter(flex_list,lcpcdi_percdiff)
ax2.set_title('lcpcdi_percdiff')
plt.tight_layout()
plt.savefig(output+'/output_percdiff_plot.png',dpi = 600)

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

cell_text = [(sindri_mean),(lcpcdi_mean)]
row_headers = ['sindri_mean','lcpcdi_mean']
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
plt.savefig(output+'/mean_table.png',
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

cell_text = [(sindri_stdev),(lcpcdi_stdev)]
row_headers = ['sindri_stdev','lcpcdi_stdev']
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
plt.savefig(output+'/stdev_table.png',
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

cell_text = [(sindri_percdiff),(lcpcdi_percdiff)]
row_headers = ['sindri_percdiff','lcpcdi_percdiff']
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
plt.savefig(output+'/percdiff_table.png',
            bbox='tight',
            edgecolor=fig.get_edgecolor(),
            facecolor=fig.get_facecolor(),
            dpi=600
            )
