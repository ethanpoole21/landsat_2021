import cv2
import matplotlib.pyplot as plt

shape_img = cv2.imread('30.tif',-1)

print(shape_img)

plt.imshow(shape_img)
plt.show()