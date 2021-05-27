import lib
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage.interpolation import rotate
from skimage.measure import label
from skimage.filters import threshold_otsu
from skimage.color import rgb2gray
import cv2
from skimage.feature import match_template
from collections import Counter




def crop_suite (card):
    card_copy=card.copy()
#print(card1_0)
    grayscale = rgb2gray(card_copy)

    thresh = threshold_otsu(grayscale)

    binary = grayscale< thresh

    labels, _ = label(binary, return_num=True, connectivity=2)
    #print(labels[60,60])
    #area_w = 20:100
    #area_h = [20:110]
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
    #print(xmin,xmax,ymin,ymax)
    cropped_suite = card[xmin:xmax,ymin:ymax]


    return cropped_suite,xmin,xmax,ymin,ymax



def detect_color(xmin,xmax,ymin,ymax,card):
    x_mean = round((xmin+xmax)/2)
    y_mean = round((ymin+ymax)/2)
    
    color = None

    if all(card[x_mean,y_mean]>[160,40,40]) and all(card[x_mean,y_mean]<[190,60,60]) :
        #print("red")
        color = 'red'

    if all(card[x_mean,y_mean]>[0,0,0]) and all(card[x_mean,y_mean]<[50,50,50]) :
        #print("black")
        color = 'black'
    
    return color

def affin_transform_suites(cropped_suite,xmin,xmax,ymin,ymax):
    
    #plt.imshow(cropped_suite)

    plt.show()
    height,width,_ = np.shape(cropped_suite)

    input_pts = np.float32([[0,0],[width,height],[0,height]])
    cols=46
    rows=63
    
    output_pts = np.float32([[0,0],[cols,rows],[0,rows]])
    
    M= cv2.getAffineTransform(input_pts , output_pts)
    res = cv2.warpAffine(cropped_suite, M, (cols,rows))

    """
    grayscale = rgb2gray(res)

    thresh = threshold_otsu(grayscale)

    binary = grayscale< thresh
    """
    return res

def binary_transform(res):
    grayscale = rgb2gray(res)

    thresh = threshold_otsu(grayscale)

    binary = grayscale< thresh

    return binary

######### creation of template
"""
# 1. find a card for each suite
suite_model_cards={} # store 4 cards 4 different suite
suite_tab = ['diamond','heart','club','spade']

file1 = "train_games/game1/3.jpg"
image1 = lib.read_image(file1)
image_dealer1 = lib.mask_dealer(image1)
extracted1 = lib.mask_for_extract_cards(image_dealer1)
players,centers,points_f,labels = lib.players_clustering(extracted1)
cards1,left_lowers,right_lowers,right_uppers = lib.find_corners(players,centers,image1)


file2 = "train_games/game1/2.jpg"
image2 = lib.read_image(file2)
image_dealer2 = lib.mask_dealer(image2)
extracted2 = lib.mask_for_extract_cards(image_dealer2)
players,centers,points_f,labels = lib.players_clustering(extracted2)
cards2,left_lowers,right_lowers,right_uppers = lib.find_corners(players,centers,image2)



# 2. Collect them
suite_model_cards['diamond'] = cards1[0]
suite_model_cards['heart'] = cards1[1]
suite_model_cards['club'] = cards1[2]
suite_model_cards['spade'] = cards2[0]


template = {}
# 3.  creation of template for all suites
for i in range(4):
    cropped_suite,xmin,xmax,ymin,ymax = crop_suite(suite_model_cards[suite_tab[i]]) # need return
    #plt.imshow(cropped_suite)
    res = affin_transform_suites(cropped_suite,xmin,xmax,ymin,ymax)
    template[suite_tab[i]] = res
    # 4 dectect color
    color = detect_color(xmin,xmax,ymin,ymax,suite_model_cards[suite_tab[i]])

for i in range(4):
    plt.subplot(1,4,i+1)
    plt.imshow(template[suite_tab[i]])
    cv2.imwrite('template_'+ suite_tab[i]+'.jpg', template[suite_tab[i]] )
plt.show()

#plt.show()
"""

"""


template_dict={}
template_list = lib.walkFile('yuan//templates')
for file in template_list:
    if 'club' in file:
        template_dict['club'] = binary_transform(lib.read_image(file))
    if 'diamond' in file:
        template_dict['diamond'] = binary_transform(lib.read_image(file))
    if 'heart' in file:
        template_dict['heart'] = binary_transform(lib.read_image(file))
    if 'spade' in file:
        template_dict['spade'] = binary_transform(lib.read_image(file))

        
suite_tab = ['diamond','heart','club','spade']     
for i in range(4):
    plt.subplot(1,4,i+1)
    plt.imshow(template_dict[suite_tab[i]])
plt.show()


file = "train_games/game4/8.jpg"
image = lib.read_image(file)
image_dealer = lib.mask_dealer(image)
plt.imshow(image_dealer)
plt.show()
extracted = lib.mask_for_extract_cards(image_dealer)
plt.imshow(extracted)
plt.show()
players,centers,points_f,labels = lib.players_clustering(extracted)
cards,left_lowers,right_lowers,right_uppers = lib.find_corners(players,centers,image)
#print(left_lowers,right_lowers,right_uppers)
for i in range(4):
    plt.subplot(1,4,i+1)
    plt.imshow(cards[i])
    plt.title('player '+ str(i+1) )
plt.show()

for i in range(4):
    cropped_suite,xmin,xmax,ymin,ymax = crop_suite(cards[i])
    color = detect_color(xmin,xmax,ymin,ymax,cards[i])
    res = affin_transform_suites(cropped_suite,xmin,xmax,ymin,ymax) # affin transform with h=63, w=46
    bin_res = binary_transform(res)
    
    best_score = 0.0
    best_index = None
    for j in range(4):
        result = match_template(bin_res, template_dict[suite_tab[j]])
        result = abs(result)
        if result > best_score:
            best_score = result
            best_index = j
    print("player: ", i+1)
    print("matching suite: ", suite_tab[best_index])
    plt.imshow(bin_res)
plt.show()

"""