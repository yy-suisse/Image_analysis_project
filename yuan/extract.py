

import os
import skimage.io
import matplotlib.pyplot as plt
import numpy as np
import os
from skimage.morphology import closing,square
from skimage.color import rgb2gray
from skimage import data, color
from skimage.transform import hough_circle, hough_circle_peaks
from skimage.util import invert
from skimage.feature import canny
from skimage.draw import circle_perimeter
from skimage.transform import hough_ellipse

def walkFile(file):
    file_list =[]
    for root, dirs, files in os.walk(file):
        for f in files:

            #print(os.path.join(root, f))
            file_list.append(os.path.join(root, f))
    return file_list




def check_green_filter_for_all(file_list):
        for single_image in file_list:
                filename, file_extension = os.path.splitext(single_image)
                if file_extension == ".jpg":

                        data1_1 = skimage.io.imread(single_image)#train_games\game1\1.jpg
                        data1_1_copy = np.copy(data1_1)
                        filter_r = np.logical_and(data1_1_copy[:,:,0] > 0.3* data1_1_copy[:,:,1],data1_1_copy[:,:,0] < 0.8*data1_1_copy[:,:,1])
                        filter_b = np.logical_and(data1_1_copy[:,:,2] > 0.3*data1_1_copy[:,:,1],data1_1_copy[:,:,2] < 0.8*data1_1_copy[:,:,1])
                        filter_green = np.logical_and(filter_r,filter_b)
                        filter_green_inv =  np.logical_not(filter_green)
                        data1_1_copy[filter_green] = [0,0,0]
                        data1_1_copy[filter_green_inv] = [255,255,255]



                        plt.imshow(data1_1_copy)
                        plt.show()



#file_list = walkFile("train_games")
#check_green_filter_for_all(file_list)

# TRY ON ONE IMAGE:---------------------------------------------------------

data1_1 = skimage.io.imread('train_games/game1/1.jpg') #train_games\game1\1.jpg
data1_1_copy = np.copy(data1_1)
filter_r = np.logical_and(data1_1_copy[:,:,0] > 0.3* data1_1_copy[:,:,1],data1_1_copy[:,:,0] < 0.7*data1_1_copy[:,:,1])
filter_b = np.logical_and(data1_1_copy[:,:,2] > 0.3*data1_1_copy[:,:,1],data1_1_copy[:,:,2] < 0.7*data1_1_copy[:,:,1])
filter_green = np.logical_and(filter_r,filter_b)
filter_green_inv =  np.logical_not(filter_green)
data1_1_copy[filter_green] =  [255,255,255]
data1_1_copy[filter_green_inv] =[0,0,0]

selem = square(30)

# change in grayscale:
grayscale_data = rgb2gray(data1_1_copy)
data1_1_copy_closed = closing(grayscale_data,selem)
plt.imshow(data1_1_copy_closed)


data1_1_copy_closed= invert(data1_1_copy_closed)
data1_1_copy_closed= color.rgb2gray(data1_1_copy_closed)

edges = canny(data1_1_copy_closed,sigma=3)

hough_radii = np.arange(200, 400, 20)
result = hough_circle(edges,hough_radii)
accums, cx, cy, radii = hough_circle_peaks(result, hough_radii,
                                           total_num_peaks=1)





print("certcle center: ", cx,cy)
print("certcle raduis: ", radii)



# draw it:
data1_1_copy_draw_cercle = np.copy(data1_1)
fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(10, 4))
for center_y, center_x, radius in zip(cy, cx, radii):
    circy, circx = circle_perimeter(center_y, center_x, radius,
                                    shape=data1_1_copy_closed.shape)
    data1_1_copy_draw_cercle[circy, circx] = [220, 20, 20]

ax.imshow(data1_1_copy_draw_cercle)
plt.show()
