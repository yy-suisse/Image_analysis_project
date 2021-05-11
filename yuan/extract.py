

import os
import skimage.io
import matplotlib.pyplot as plt
import numpy as np
import os

def walkFile(file):
    file_list =[]
    for root, dirs, files in os.walk(file):
        for f in files:

            #print(os.path.join(root, f))
            file_list.append(os.path.join(root, f))
    return file_list



file_list = walkFile("train_games")

for single_image in file_list:
        filename, file_extension = os.path.splitext(single_image)
        if file_extension == ".jpg":

                data1_1 = skimage.io.imread(single_image)#train_games\game1\1.jpg
                data1_1_copy = np.copy(data1_1)
                #filter_green = np.logical_and(data1_1_copy[:,:,1]<110,data1_1_copy[:,:,1]>90)
                filter_r = np.logical_and(data1_1_copy[:,:,0] > 0.3* data1_1_copy[:,:,1],data1_1_copy[:,:,0] < 0.7*data1_1_copy[:,:,1])
                filter_b = np.logical_and(data1_1_copy[:,:,2] > 0.3*data1_1_copy[:,:,1],data1_1_copy[:,:,2] < 0.7*data1_1_copy[:,:,1])
                filter_green = np.logical_and(filter_r,filter_b)
                filter_green_inv =  np.logical_not(filter_green)
                data1_1_copy[filter_green] = [0,0,0]
                data1_1_copy[filter_green_inv] = [255,255,255]

                #data1_1_copy[data1_1[:,:,0]<upper_green] = 0
                #data1_1_copy[data1_1[:,:,:]>lower_green] = 0


                plt.imshow(data1_1_copy)
                plt.show()



