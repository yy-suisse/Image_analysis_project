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
    """
    This function receives one card, use the object labeling to find its suit location (upper left part of the card) and 
    compute the region of suit
    Args: 
        card: one card isolated from the image 
    Return: 
        cropped_suite: the suit cropped from the card
        xmin: the minimum value of coordinate on x for the suit on the current image
        xmax: the maximum value of coordinate on x for the suit on the current image
        ymin: the minimum value of coordinate on y for the suit on the current image
        ymax: the maximum value of coordinate on y for the suit on the current image
    """
    
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
    """
    This function receives one card and the range of coordinate on x and y for its upper left suit, then compute 
    the color of this suit by looking at the color of the pixel in the middle of suit region
    Args: 
        xmin: the minimum value of coordinate on x for the suit on the current image
        xmax: the maximum value of coordinate on x for the suit on the current image
        ymin: the minimum value of coordinate on y for the suit on the current image
        ymax: the maximum value of coordinate on y for the suit on the current image
        card: one card isolated from the image 
    Return: 
        color: the color of the suit
        
    """
    x_mean = round((xmin+xmax)/2)
    y_mean = round((ymin+ymax)/2)
    
    color = None

    if all(card[x_mean,y_mean]>=[140,20,20]) and all(card[x_mean,y_mean]<=[210,80,80]) :
        color = 'red'

    if all(card[x_mean,y_mean]>[0,0,0]) and all(card[x_mean,y_mean]<[100,100,100]) :
        color = 'black'
    
    return color

def affin_transform_suites(cropped_suite,xmin,xmax,ymin,ymax):
    """
    This function applys an affin transform on the cropped suit in order to make it fit into a certain size 
    and orientation.
    
    Args: 
        cropped_suite: the suit cropped from the card
        xmin: the minimum value of coordinate on x for the suit on the current image
        xmax: the maximum value of coordinate on x for the suit on the current image
        ymin: the minimum value of coordinate on y for the suit on the current image
        ymax: the maximum value of coordinate on y for the suit on the current image
        
    Return: 
        res: the suit figure after the affin transformation (63*46 pixels)
        
    """

    height,width,_ = np.shape(cropped_suite)

    input_pts = np.float32([[0,0],[width,height],[0,height]])
    cols=46
    rows=63
    
    output_pts = np.float32([[0,0],[cols,rows],[0,rows]])
    
    M= cv2.getAffineTransform(input_pts , output_pts)
    res = cv2.warpAffine(cropped_suite, M, (cols,rows))

    return res

def binary_transform(res):
    """
    This function convert a RGB image into a binary image by using grayscale and thresholding method
    Args:
        res: a RGB image
    Return:
        binary: a bnary image
    """
    grayscale = rgb2gray(res)

    thresh = threshold_otsu(grayscale)

    binary = grayscale< thresh

    return binary

def show_suites_for_all_players(cards,template_dict):
    """
    This function receives four cards of one image, preprocess to isolate the suits for four cards, then apply the 
    template matching method between the suits isolated from the cards and the suit models in the template_dict
    to classify the suits
    Arg:
        cards: four cards (with same size and good orientation) extracted from one image
        template_dict: a list containing four suit models, one for each
        
    Return: 
        suites: a list containing the best fitted suits for all four cards
    """
    
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
    """
    This function accesses to the folder and read four suit templates 
    Arg:
        folder: path of folder which stores the four suit template models
        
    Return: 
        template_dict: a dictionary which store all the four template models
    """
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
