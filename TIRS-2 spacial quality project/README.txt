Project Title:

TIRS-2 spacial quality project

Description:

This code will generate the post- launch response analysis of TIRS-2 thermal images and return metrics that quantitatively assess the spatial quality of the images

How to run:

First with Earth Explorer or another chosen database select images that include a vertical (on track) or horizontal (off track) ‘thermal knife edge’ where 
a large body of water intersects a desert edge. With these images a ROI is selected by running the find_points.py code or with the simplified version lightdark.py. 
next using the psf_mtf.ipynb file (will display step by step method) or psfmtf.py ( all in one) run analysis of image. 

Output:

The output should be a plot of the edge spread function as sell as calculations of the full width half max, the spectral resolution, and the signal to noise ratio of the system

Authors:

Ethan Poole 
ethan(dot)r(dot)poole(at)gmail(dot)com

Version:

1.0.0

License:

GPL-3.0-or-later

Acknowledgments:

Wenny, B.N.; Helder, D.; Hong, J.; Leigh, L.; Thome, K.J.; Reuter, D. Pre- and Post-Launch Spatial 
Quality of the Landsat 8 Thermal Infrared Sensor. Remote Sens. 2015, 7, 1962-1980. https://doi.org/10.3390/rs70201962