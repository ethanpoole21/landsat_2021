import os
import sys
import argparse
import numpy as np
import PIL 
from PIL import Image
import csv
from statistics import stdev
import glob
import rasterio
import matplotlib.pyplot as plt
from envi_header import find_hdr_file,read_hdr_file,write_envi_header
import spectral.io.envi as envi
from colorimg import *
from io import BytesIO
import cv2

# images
rad_dir = '1916rad'
temp_dir = '1916temp'
output = '1916_rois'
filenames = ['1916_out/roi1_averages.csv','1916_out/roi2_averages.csv','1916_out/roi3_averages.csv','1916_out/roi4_averages.csv','1916_out/roi5_averages.csv','1916_out/roi6_averages.csv']
time_headers = '1916rad'

rad_paths = glob.glob(rad_dir + '/*.img')
rad_paths.sort()
rad_headers = glob.glob(rad_dir + '/*.hdr')
rad_headers.sort()

temp_paths = glob.glob(temp_dir + '/*.img')
temp_headers = glob.glob(temp_dir + '/*.hdr')

#multiple_length = len(rad_paths)
#print(multiple_length)

shape_img = rasterio.open(rad_paths[0])
shape_img = shape_img.read()

dims =np.asarray(shape_img).shape

r = dims[1]
c = dims[2]
q = r*c

n = 0

rows = []

for path in rad_paths:

    rois = []
    
    imname = str(path.split("/",1)[1])
    imname = str(imname.split(".",1)[0])

    print(imname)

    corr_temp = [sub for sub in temp_paths if all(ele in sub for ele in imname)]
    corr_head = [sub for sub in rad_headers if all(ele in sub for ele in imname)]

    corr_temp = temp_dir +'/'+ imname + '.img'
    corr_head = time_headers +'/'+ imname + '.hdr'

    rad_image = rasterio.open(path)
    rad_image = rad_image.read(1).astype(np.float32)

    corr_temp = str(corr_temp)
    corr_head = str(corr_head)

    print(corr_head)

    temp_image = rasterio.open(corr_temp)
    temp_image = temp_image.read(1).astype(np.float32)

    #curr_head = envi.open(corr_head)
    curr_head = read_hdr_file(corr_head)
    tuple_list = list(curr_head.items())
    time = tuple_list[10]
    print(time)

    flat_rad_image = rad_image.flat
    fin_radiances = np.asarray(flat_rad_image).reshape(r,c)

    flat_temp_image = temp_image.flat
    fin_temps = np.asarray(flat_temp_image).reshape(r,c)

    rois = cv2.selectROIs("Select Rois",rad_image.astype(np.uint8))

    color = (100, 100, 100)
    thickness = int(1)

    num = 1

    roi_im = np.zeros([r, c])

    for roi in rois:

        row = []

        start_point = int(roi[0]),int(roi[1])
        end_point = int(roi[0]+roi[2]),int(roi[1]+roi[3])
        roi_im = cv2.rectangle(roi_im, start_point, end_point, color, thickness)
        org = (end_point[0], end_point[1])
        thick = 1
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = .5
        roi_im = cv2.putText(roi_im, str(num), org, font, fontScale, color, thick)

        rad_cropped = fin_radiances[int(roi[1]):int(roi[1]+roi[3]), int(roi[0]):int(roi[0]+roi[2])]
        temp_cropped = fin_temps[int(roi[1]):int(roi[1]+roi[3]), int(roi[0]):int(roi[0]+roi[2])]

        rad_average = np.average(rad_cropped)
        rad_stdev = np.std(rad_cropped)

        temp_average = np.average(temp_cropped)
        temp_stdev = np.std(temp_cropped)

        row.append(imname)
        row.append(start_point)
        row.append(end_point)
        row.append(rad_average)
        row.append(rad_stdev)
        row.append(temp_average)
        row.append(temp_stdev)
        row.append(time)

        num +=1

        rows.append(row)

    print(row)

    plt.imshow(rad_image,cmap=plt.get_cmap('gray'),vmin = np.percentile(rad_image,2), vmax = np.percentile(rad_image,98))
    plt.imshow(roi_im, alpha=.5)
    plt.title(imname)
    plt.savefig(output+'/roi_locations'+str(n)+'.png')
    plt.show()

    n+=1


fields = ['Im_name','x_coordinate', 'y_coordinate', 'mean_radiance','stdev_radiance','mean_temperature','stdev_temperature','time']


for filename in filenames:
    with open(filename, 'w') as csvfile:  
        csvwriter = csv.writer(csvfile) 
        csvwriter.writerow(fields) 
        csvfile.close()

e = 0
for row in rows:
    with open(filenames[e], 'a+', newline='') as csvfile: 
        csvwriter = csv.writer(csvfile) 
        csvwriter.writerow(row)
    if e < 5:
        e +=1
    else:
        e = 0
