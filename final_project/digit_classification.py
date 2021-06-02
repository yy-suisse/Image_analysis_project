from utils import evaluate_game
import skimage.io
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from pathlib import Path
from skimage import color
from skimage import io
import cv2 as cv
import matplotlib.image as mpimg
from skimage.transform import resize

import gzip
import numpy as np
import matplotlib.pyplot as plt
import os

from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.ensemble import BaggingClassifier

def generate_ordered_cgt_mask(cgt_rank):
    """
    This function generated ordered number including jqk of cards(game,player,round) and masks that indicates whether the central symbol is number or jqk 
    Args: 
        cgt_rank: suites and central symbols extracted from each game
    Return: 
        cgt_order: ordered array of central symbols in one game
        cgt_mask_order: 0&1 array that indicates whether the central symbol is number or jqk
    """
    card_dict={'0':0,'1':1,'2':2,'3':3,'4':4,'5':5,'6':6,'7':7,'8':8,'9':9,'J':10,'Q':11,'K':12}
    num_dict={'0':1,'1':1,'2':1,'3':1,'4':1,'5':1,'6':1,'7':1,'8':1,'9':1,'J':0,'Q':0,'K':0}
    cgt_number=np.zeros((13,4), dtype=int)
    cgt_mask=np.zeros((13,4), dtype=int)
    i=0
    j=0
    for n in cgt_rank:
        i=i+1
        j=0
        for e in n:
            j=j+1
            #print(card_dict[e[0]])
            cgt_number[i-1][j-1]=card_dict[e[0]]
            cgt_mask[i-1][j-1]=num_dict[e[0]]
    cgt_order=cgt_number.transpose().reshape(-1).tolist()
    cgt_mask_order=cgt_mask.transpose().reshape(-1).tolist()
    return cgt_order, cgt_mask_order

def zerolistmaker(n):
    """
    This function generated full zeros list
    Args: 
        n: integer that is length of list
    Return: 
        listofzeros: list full of zeros
    """
    listofzeros = [0] * n
    return listofzeros


def rgb2gray(rgb):
    """
    This function converts rgb images into grayscale images
    Args: 
        rgb: rgb images
    Return: 
        image: grayscale images
    """
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140]) 

def sorter(item):
    """
    This function defines sorter to facilitate sorting of extracted central symbols.
    Args: 
        item: name of extracted central symbols.
    Return: 
        tuple: sorting order of game, player and round.
    """
    order_dict={'1':1,'2':2,'3':3,'4':4,'5':5,'6':6,'7':7,'8':8,'9':9,'10':10,'11':11,'12':12,'13':13}
    g_p_l=item.split('_')
    game =g_p_l[0] 
    player= g_p_l[1]
    loop= g_p_l[2]
    loop_n=loop.split('.')
    return (game, player,order_dict[loop_n[0]])

def extract_data(filename, image_shape, image_number):
    """
    This function extracts images from MNIST dataset of certain number.
    Args: 
        filename: path of dataset
        image_shape: 28*28 images
        image_number: number of extracted images
    Return: 
        data: (image_number,image_shape) array
    """
    with gzip.open(filename) as bytestream:
        bytestream.read(16)
        buf = bytestream.read(np.prod(image_shape) * image_number)
        data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
        data = data.reshape(image_number, image_shape[0], image_shape[1])
    return data


def extract_labels(filename, image_number):
    """
    This function extracts corresponding labels of images from MNIST dataset of certain number.
    Args: 
        filename: path of dataset
        image_number: number of extracted images
    Return: 
        data: (image_number) array
    """
    with gzip.open(filename) as bytestream:
        bytestream.read(8)
        buf = bytestream.read(1 * image_number)
        labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
    return labels

def NN_for_classification(x_train, y_train):
    """
    This function generates two neural network for classification tasks, one is MLPs and the other MLPs with bagging as ensemble learning
    Args: 
        x_train: input of neural networks for training
        y_train: labels of input for training
    Return: 
        MLP_clf: MLPs networks for classification
        Ensemble_clf: MLPs with bagging as ensemble learning for classification
    """
    #input the train data/label, test data/label, output MLP NN and ensemble NN
    MLP_clf = MLPClassifier(hidden_layer_sizes=250)
    Ensemble_clf= BaggingClassifier(base_estimator=SVC(),n_estimators=10, random_state=0)
    # training the MLP
    MLP_clf.fit(x_train, y_train)
    Ensemble_clf.fit(x_train, y_train)
    return MLP_clf, Ensemble_clf

def digit_extract(card):
    """
    This function transform extracted central cards into images that can be used as training samples
    Args: 
        card: extracted images of central cards
    Return: 
        th_im: processed images(rgb to grayscale, crop and interpolation, binary images with adaptive threshold)
    """
    gray_card = rgb2gray(card)
    card_crop_gray = cv.resize(gray_card[110:290,70:250], (28, 28),interpolation=cv.INTER_AREA)
    card_crop_gray = card_crop_gray.astype(np.uint8)
    th_im = cv.adaptiveThreshold(card_crop_gray,255,cv.ADAPTIVE_THRESH_MEAN_C,\
                cv.THRESH_BINARY,7,2)
    th_im = 255 - th_im
    return th_im

def preprocessing_for_cards():
    """
    This function process the extracted central cards and and symbols and generate two training sets(one for digits classification, the other for classification of all symbols)
    Args: 
        None
    Return: 
        (x_train_0, y_train_0): training set for digits classification
        (x_train_1, y_train_1): training set for classification of all card central symbols 
    """
    card_dict={'0':0,'1':1,'2':2,'3':3,'4':4,'5':5,'6':6,'7':7,'8':8,'9':9,'J':10,'Q':11,'K':12}
    order_dict={'1':1,'2':2,'3':3,'4':4,'5':5,'6':6,'7':7,'8':8,'9':9,'10':10,'11':11,'12':12,'13':13}
    num_dict={'0':1,'1':1,'2':1,'3':1,'4':1,'5':1,'6':1,'7':1,'8':1,'9':1,'J':0,'Q':0,'K':0}
    #### extract labels 
    label_card=[]
    mask_card=[]
    for i in range(6):
        j=i+1
        cgt = pd.read_csv('train_games/game'+str(j)+'/game'+str(j)+'.csv', index_col=0)
        cgt_rank = cgt[['P1', 'P2', 'P3', 'P4']].values
        cgt_order, cgt_mask_order=generate_ordered_cgt_mask(cgt_rank)
        label_card=label_card+cgt_order
        mask_card=mask_card+cgt_mask_order
    label_card_test=[]
    mask_card_test=[]
    cgt_t = pd.read_csv('train_games/game'+str(7)+'/game'+str(7)+'.csv', index_col=0)
    cgt_rank_t = cgt_t[['P1', 'P2', 'P3', 'P4']].values
    #print(cgt_rank_t)
    label_card_test, mask_card_test=generate_ordered_cgt_mask(cgt_rank_t)
    ######
    label_card_jqk=label_card.copy() ## the first 6 games label with jqk
    mask_card_jqk=mask_card.copy()  ## the mask of the first 6 games label with jqk
    label_card_test_jqk=label_card_test.copy() # the last game label with jqk
    mask_card_test_jqk=mask_card_test.copy()# the last game mask label with jqk
    ######

    for i in range(13*4*6):
        if mask_card[i]==0:
            label_card[i]='d'
    while True:
        try:
            label_card.remove('d')
        except ValueError:
            break

    for i in range(13*4):
        if mask_card_test[i]==0:
            label_card_test[i]='d'
    while True:
        try:
            label_card_test.remove('d')
        except ValueError:
            break
    
    ##preprocessing for images below
    digit_path = os.path.join('cards_latest')
    #print(digit_path)
    digit_names = [nm for nm in os.listdir(digit_path) if '.jpg' in nm]  # make sure to only load .png
    sorted_names = sorted(digit_names, key=sorter)
    ic = skimage.io.imread_collection([os.path.join(digit_path, nm) for nm in sorted_names])
    digit_im = skimage.io.concatenate_images(ic)
    img_crop_gray=np.zeros((4*13*7,180,180)) 
    img_resize_test=np.zeros((4*13*7,28,28))
    ad_thre_test=np.zeros((4*13*7,28,28)) 
    for i in range(28):
        for j in range(13):
            im=digit_im[13*i+j]
            nm=sorted_names[13*i+j]
            gray_im = rgb2gray(im) 
            img_crop_gray[13*i+j]=gray_im[110:290,70:250]
            img_resize_test[13*i+j]=cv.resize(gray_im[110:290,70:250], (28, 28),interpolation=cv.INTER_AREA)
            inter_img=img_resize_test[13*i+j]
            inter_img= inter_img.astype(np.uint8)
            th_im = cv.adaptiveThreshold(inter_img,255,cv.ADAPTIVE_THRESH_MEAN_C,\
                cv.THRESH_BINARY,7,2)
            th_im=255-th_im
            ad_thre_test[13*i+j]=th_im
### extract test set data
    train_card_set=ad_thre_test[0:13*4*6]
    train_card_digit=train_card_set[np.array(mask_card)==1,:,:]

    train_card_jqk=train_card_set

    test_card_set=ad_thre_test[13*4*6:13*4*7]
    test_card_set_digit=test_card_set[np.array(mask_card_test)==1,:,:]

    test_card_jqk=test_card_set
##convert list to array
    label_card=np.array(label_card)
    label_card_test=np.array(label_card_test)
    label_card_jqk=np.array(label_card_jqk)
    label_card_test_jqk=np.array(label_card_test_jqk)
## extract data from MNIST library
    image_shape = (28, 28)
    train_set_size = 2000
    test_set_size=1000

    data_folder = os.path.join("mnist_data")

    train_images_path = os.path.join(data_folder, 'train-images-idx3-ubyte.gz')
    train_labels_path = os.path.join(data_folder, 'train-labels-idx1-ubyte.gz')
    test_images_path = os.path.join(data_folder, 't10k-images-idx3-ubyte.gz')
    test_labels_path = os.path.join(data_folder, 't10k-labels-idx1-ubyte.gz')

    train_images = extract_data(train_images_path, image_shape, train_set_size)
    test_images = extract_data(test_images_path, image_shape, test_set_size)
    train_labels = extract_labels(train_labels_path, train_set_size)
    test_labels = extract_labels(test_labels_path, test_set_size)
##generate two tuples for digit classification and jqk plus digits classification
    x_train_0=np.concatenate((train_images,train_card_digit),axis=0) # mix the data from MNIST with extracted and processed digits from cards
    size_train_0 = x_train_0.shape[0]
    x_train_0= np.reshape(x_train_0, [size_train_0, 784])
    y_train_0=np.concatenate((train_labels,label_card),axis=0)
    x_train_1=train_card_jqk
    size_train_1 = x_train_1.shape[0]
    x_train_1= np.reshape(x_train_1, [size_train_1, 784])
    y_train_1=label_card_jqk
    
    return (x_train_0, y_train_0),(x_train_1, y_train_1)
