
import glob
import rasterio
import matplotlib.pyplot as plt
from envi_header import find_hdr_file,read_hdr_file,write_envi_header
import spectral.io.envi as envi
import cv2
import numpy as np

planc = 'invplanc_images'
lut = 'lut_images'
temp = 'temp_images'

planc_paths = glob.glob(planc + '/*.tif')
print(len(planc_paths))
lut_paths = glob.glob(lut + '/*.tif')

temp_paths = glob.glob(temp + '/*.tif')


for e in range(1):
    output = 'roi2'
    current_img = (cv2.imread(planc_paths[e],-1)).astype(np.float32)
    print(current_img.dtype)
    roi = cv2.selectROI(current_img.astype(np.uint8))
    print(roi)
    color = (0, 0, 0)
    thickness = int(2)
    start_point = int(roi[0]),int(roi[1])
    end_point = int(roi[0]+roi[2]),int(roi[1]+roi[3])
    image = cv2.rectangle(current_img, start_point, end_point, color, thickness)
    plt.imshow(image,cmap=plt.get_cmap('gray'),vmin = np.percentile(image,2), vmax = np.percentile(image,98))
    plt.colorbar()
    plt.savefig(output+'/roi_location'+'.png')
    plt.show()
    

    current_img = (cv2.imread(planc_paths[e],-1)).astype(np.float32)
    plt.imshow(current_img, cmap=plt.get_cmap('gray'),vmin = np.percentile(current_img,2), vmax = np.percentile(current_img,98))
    plt.colorbar()
    plt.show()
    roi_cropped = current_img[int(roi[1]):int(roi[1]+roi[3]), int(roi[0]):int(roi[0]+roi[2])]
    plt.imshow(roi_cropped, cmap=plt.get_cmap('gray'),vmin = np.percentile(roi_cropped,2), vmax = np.percentile(roi_cropped,98))
    plt.colorbar()
    plt.savefig(output+'/roi_planc'+'.png')
    plt.show()
    average = np.average(roi_cropped)
    print('planc average:',average)

    current_img = (cv2.imread(lut_paths[e],-1)).astype(np.float32)
    plt.imshow(current_img, cmap=plt.get_cmap('gray'),vmin = np.percentile(current_img,2), vmax = np.percentile(current_img,98))
    plt.colorbar()
    plt.show()
    roi_cropped = current_img[int(roi[1]):int(roi[1]+roi[3]), int(roi[0]):int(roi[0]+roi[2])]
    plt.imshow(roi_cropped, cmap=plt.get_cmap('gray'),vmin = np.percentile(roi_cropped,2), vmax = np.percentile(roi_cropped,98))
    plt.colorbar()
    plt.savefig(output+'/roi_lut'+'.png')
    plt.show()
    average = np.average(roi_cropped)
    print('lut average:',average)

    current_img = (cv2.imread(temp_paths[e],-1)).astype(np.float32)
    plt.imshow(current_img, cmap=plt.get_cmap('gray'),vmin = np.percentile(current_img,2), vmax = np.percentile(current_img,98))
    plt.colorbar()
    plt.show()
    roi_cropped = current_img[int(roi[1]):int(roi[1]+roi[3]), int(roi[0]):int(roi[0]+roi[2])]
    plt.imshow(roi_cropped, cmap=plt.get_cmap('gray'),vmin = np.percentile(roi_cropped,2), vmax = np.percentile(roi_cropped,98))
    plt.colorbar()
    plt.savefig(output+'/roi_temp'+'.png')
    plt.show()
    average = np.average(roi_cropped)
    print('temp average:',average)