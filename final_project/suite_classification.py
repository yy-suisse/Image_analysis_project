# -*- coding: utf-8 -*-
import lib,cards_segmentation
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage.interpolation import rotate
from skimage.measure import label
from skimage.filters import threshold_otsu
from skimage.color import rgb2gray
import cv2
from skimage.feature import match_template
from collections import Counter
from scipy import ndimage
import os

def crop_suite (card):
    card_copy=card.copy()

    grayscale = rgb2gray(card_copy)

    thresh = threshold_otsu(grayscale)

    binary = grayscale< thresh

    labels, _ = label(binary, return_num=True, connectivity=2)
    label_list = []
    
    for i in range(20,100):
        for j in range(20,110):
            if labels[i,j] != 0:
                label_list.append(labels[i,j])
    
    occurence_count = Counter(label_list)
    res=occurence_count.most_common(1)[0][0]
    
    seed_label = res
    x_range,y_range=np.shape(labels)

    xmin = 500
    ymin = 500
    xmax = 0
    ymax = 0
    for i in range(x_range):
        for j in range(y_range):
            if labels[i,j] == seed_label:
                if i < xmin:
                    xmin = i
                if i > xmax:
                    xmax = i
                if j < ymin:
                    ymin = j
                if j > ymax:
                    ymax = j

    cropped_suite = card[xmin:xmax,ymin:ymax]

    return cropped_suite,xmin,xmax,ymin,ymax



def detect_color(xmin,xmax,ymin,ymax,card):
    x_mean = round((xmin+xmax)/2)
    y_mean = round((ymin+ymax)/2)
    
    color = None

    if all(card[x_mean,y_mean]>=[140,20,20]) and all(card[x_mean,y_mean]<=[210,80,80]) :
        color = 'red'

    if all(card[x_mean,y_mean]>[0,0,0]) and all(card[x_mean,y_mean]<[100,100,100]) :
        color = 'black'
    
    return color

def affin_transform_suites(cropped_suite,xmin,xmax,ymin,ymax):

    plt.show()
    height,width,_ = np.shape(cropped_suite)

    input_pts = np.float32([[0,0],[width,height],[0,height]])
    cols=46
    rows=63
    
    output_pts = np.float32([[0,0],[cols,rows],[0,rows]])
    
    M= cv2.getAffineTransform(input_pts , output_pts)
    res = cv2.warpAffine(cropped_suite, M, (cols,rows))

    return res

def binary_transform(res):
    grayscale = rgb2gray(res)

    thresh = threshold_otsu(grayscale)

    binary = grayscale< thresh

    return binary

def show_suites_for_all_players(cards,template_dict):
    
    suites = []

    for i in range(4):
        cropped_suite,xmin,xmax,ymin,ymax = crop_suite(cards[i])
        
        res = affin_transform_suites(cropped_suite,xmin,xmax,ymin,ymax) # affin transform with h=63, w=46
        color = detect_color(xmin,xmax,ymin,ymax,cards[i])
        bin_res = binary_transform(res)

        best_score = 0.0
        best_suite = 'diamond'
        
        possible_suites = []
        if color == 'red':
            possible_suites = ['diamond', 'heart']
        if color == 'black':
            possible_suites = ['spade', 'club']
        for j in possible_suites:
            result = match_template(bin_res, template_dict[j])
            result = abs(result)

            if result > best_score:
                best_score = result
                best_suite = j

        suites.append(best_suite)
        
    return suites
        

def load_template_suites(folder):
    template_dict={}
    template_list = cards_segmentation.walkFile(folder)
    for file in template_list:
        if 'club' in file:
            template_dict['club'] = binary_transform(cards_segmentation.read_image(file))
        if 'diamond' in file:
            template_dict['diamond'] = binary_transform(cards_segmentation.read_image(file))
        if 'heart' in file:
            template_dict['heart'] = binary_transform(cards_segmentation.read_image(file))
        if 'spade' in file:
            template_dict['spade'] = binary_transform(cards_segmentation.read_image(file))

            
    suite_tab = ['diamond','heart','club','spade']   
    
    return template_dict
