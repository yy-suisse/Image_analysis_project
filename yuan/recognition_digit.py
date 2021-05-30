import card_segmentation
import numpy as np
from scipy import ndimage
from skimage.measure import label
from skimage.filters import threshold_otsu
from skimage.color import rgb2gray
import cv2
import matplotlib.pyplot as plt


def find_king(card):
    card_copy=card.copy()
#print(card1_0)
    grayscale = rgb2gray(card_copy)

    thresh = threshold_otsu(grayscale)

    xmin = 45
    ymin = 110
    xmax = 250
    ymax = 286

    binary = grayscale< thresh


    labels, nb = label(binary[ymin:ymax,xmin:xmax], return_num=True, connectivity=2)
    #labels, nb = label(binary[:,:], return_num=True, connectivity=2)
    
    nb_effective = 0
    for i in range(1,nb+1):
        if np.sum(labels==i) >=50:
            nb_effective += 1
    if nb_effective>=3:
        print("king found")







### main  for all ###
"""
file_list = card_segmentation.walkFile("train_games")
file_pics=[]
print(file_list)
for file in file_list:
    if file.endswith('.jpg'):
        print(file)
        image = card_segmentation.read_image(file)
            
            
        mask_t = card_segmentation.mask_thre(image)
        image = card_segmentation.mask_dealer(image)
        mask_r = card_segmentation.mask_range(image)
        labeled_array, num_features = ndimage.label(mask_t, structure = np.ones((3,3)))

        for i in range(num_features):
            if np.sum(mask_r[labeled_array == i+1]) < 255 * 600:
                mask_t[labeled_array == i+1] = 0
                
        mask = np.uint8(mask_t)

        players,centers,points_f,labels = card_segmentation.players_clustering(mask)
        cards,left_lowers,right_lowers,right_uppers = card_segmentation.find_corners(players,centers,image)
        for card in cards:
            find_king(card)
"""

### main  for one ###
"""
file = 'train_games/game7/12.jpg'
image = card_segmentation.read_image(file)
    
    
mask_t = card_segmentation.mask_thre(image)
image = card_segmentation.mask_dealer(image)
mask_r = card_segmentation.mask_range(image)
labeled_array, num_features = ndimage.label(mask_t, structure = np.ones((3,3)))

for i in range(num_features):
    if np.sum(mask_r[labeled_array == i+1]) < 255 * 600:
        mask_t[labeled_array == i+1] = 0
        
mask = np.uint8(mask_t)

players,centers,points_f,labels = card_segmentation.players_clustering(mask)
cards,left_lowers,right_lowers,right_uppers = card_segmentation.find_corners(players,centers,image)
for card in cards:
    find_king(card)
"""