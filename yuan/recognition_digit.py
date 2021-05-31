import card_segmentation,find_suite
import numpy as np
from scipy import ndimage
from skimage.measure import label
from skimage.filters import threshold_otsu
from skimage.color import rgb2gray
import cv2
import matplotlib.pyplot as plt
from skimage.morphology import thin


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


def extract_card(file):
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

    return cards

def load_1_J_training(folder):
    file_list = card_segmentation.walkFile(folder)
    
    ones_im = []
    Js_im = []
    print(file_list)
    for file in file_list:
        if file.endswith('.jpg') and 'J' in file:
            image = card_segmentation.read_image(file)
            Js_im.append(image)
        else:
            image = card_segmentation.read_image(file)
            ones_im.append(image)

    return ones_im,Js_im
        
def thin_images (J_images, one_images):
    xmin = 45
    ymin = 110
    xmax = 250
    ymax = 286

    J_im_copy = J_images.copy()
    Js_thin=[]

    one_im_copy = one_images.copy()
    ones_thin=[]


    for im in J_im_copy:
        im = im[ymin:ymax,xmin:xmax]
        bin_im = find_suite.binary_transform(im)
        Js_thin.append(thin(bin_im))


       
    for im in one_im_copy:
        im = im[ymin:ymax,xmin:xmax]
        bin_im = find_suite.binary_transform(im)
        ones_thin.append(thin(bin_im))

    
    return Js_thin,ones_thin


def thin_images_test_one_card (card):

    xmin = 45
    ymin = 110
    xmax = 250
    ymax = 286

    card_copy = card.copy()
    card_copy = card_copy[ymin:ymax,xmin:xmax]
    bin_im = find_suite.binary_transform(card_copy)

    return thin(bin_im)

    


def compute_dft(img,i = 1,j = 2):  
    """
    extract contour of image, apply an transformation on the contour if 
    necessary, compute FFT, then calculate the magnitude of FFT for given frequencies.

    arg: 
      img: preprocessed image
      i,j: 2 frequencies chosen, by default 1 and 2
      rotation, scaling, translation: options for transformation, False by default.

    return:
      amplitude of FFT for the given frequencies i,j divided by scaling factor (if scaling == True)
      amplitude of FFT for the given frequencies i,j (otherwise)

    """
    img = 255 * np.array(img).astype('uint8')
    contours, hierarchy = cv2.findContours(
        img,  cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    contour_array = contours[0][:, 0, :]
    contour_complex = np.empty(contour_array.shape[:-1], dtype=complex)      # complex buffer for storing contours
    contour_complex_transformed = np.empty(contour_array.shape[:-1], dtype=complex) # complex buffer for storing transfored contours

    contour_complex.real=contours[0][:,0,0]
    contour_complex.imag=contours[0][:,0,1]
   
    
    contour_complex_transformed = contour_complex

    fourier_result = np.fft.fft(contour_complex_transformed)
    fourier_power = np.abs(fourier_result)
    

    return fourier_power[i]/fourier_power[0],fourier_power[j]/fourier_power[0]
        
def plot_result(Js_thin,ones_thin, f1, f2,rotation = None,scaling = None,translation = None):
    A1=[]
    A2=[]

    for i in range (0,8,1):
        x_1,y_1=compute_dft(Js_thin[i],f1,f2)
        A1.append(x_1)
        A2.append(y_1)
        

    A3=[]
    A4=[]
    for i in range (0,8,1):
        x_0,y_0=compute_dft(ones_thin[i],f1,f2)
        A3.append(x_0)
        A4.append(y_0)

    return A1,A2,A3,A4

def one_or_J (card,J_datas,one_datas):
    img = thin_images_test_one_card(card)
    f1,f2 = compute_dft(img,i = 1,j = 2)
    mu_J = np.mean (J_datas, axis = 0, dtype = None) 
    mu_1 = np.mean (one_datas, axis = 0, dtype = None) 

    # For simplicity reasons, we round the estimated covariant matrix to the closest integer value.
    J_datas_t = np.array(J_datas).T
    covMatrix_J = np.cov(J_datas_t,bias=True)


    one_datas_t = np.array(one_datas).T
    covMatrix_1 = np.cov(one_datas_t,bias=True)


    # computation of inverse covariant matrix 
    covMatrix_J_inv = np.linalg.inv(covMatrix_J)
    covMatrix_1_inv = np.linalg.inv(covMatrix_1)

    point=np.matrix([f1,f2])
    dist_to_J = np.sqrt((point-mu_J)*covMatrix_J_inv*np.transpose(point-mu_J))
    dist_to_1 = np.sqrt((point-mu_1)*covMatrix_1_inv*np.transpose(point-mu_1))

    dist_to_J = float(dist_to_J)
    dist_to_1 = float(dist_to_1)
    
    if dist_to_J <= dist_to_1: 
        print("this is J")
    else:
        print("this is 1")






### fourrier descriptor ###



# construct the FD classifier 
ones_im,Js_im = load_1_J_training('yuan/1_and_j')
Js_thin,ones_thin = thin_images(Js_im, ones_im)
o_J_0,o_J_1,o_1_0,o_1_1 = plot_result(Js_thin, ones_thin,1,2)



plt.figure(figsize=(18, 9)) 

plt.subplot(2, 6, 1)
plt.scatter(o_J_0,o_J_1,marker='o', s=40, c='black', alpha=0.5, label = 'J')
plt.scatter(o_1_0,o_1_1,marker='o', s=40, c='red', alpha=0.5, label = '1')
plt.legend(loc = "upper left")
plt.xlabel('f1')
plt.ylabel('f2')
plt.title('J VS 1')
plt.show()

J_datas_temp = [[a,b] for a,b in zip(o_J_0, o_J_1)]
J_datas = np.asarray(J_datas_temp)

one_datas_temp = [[a,b] for a,b in zip(o_1_0, o_1_1)]
one_datas = np.asarray(one_datas_temp)


# test card J
file_J = 'train_games/game5/1.jpg'
cards = extract_card(file_J)
card_J = cards[0]
one_or_J (card_J,J_datas,one_datas)

# test card one
file_one = 'train_games/game5/8.jpg'
cards = extract_card(file_one)
card_one = cards[1]
one_or_J (card_one,J_datas,one_datas)



### main for all FIND KING ###
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

### main  for one  FIND KING  ###
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

# EXTRACT 1 AND J
"""
cards_1 =[]

# read "1"s:
file = 'train_games/game1/3.jpg'
cards = extract_card(file)
cards_1.append(cards[0])

file = 'train_games/game1/4.jpg'
cards = extract_card(file)
cards_1.append(cards[0])

file = 'train_games/game1/9.jpg'
cards = extract_card(file)
cards_1.append(cards[2])

file = 'train_games/game1/13.jpg'
cards = extract_card(file)
cards_1.append(cards[0])

file = 'train_games/game2/4.jpg'
cards = extract_card(file)
cards_1.append(cards[1])

file = 'train_games/game2/5.jpg'
cards = extract_card(file)
cards_1.append(cards[3])

file = 'train_games/game2/8.jpg'
cards = extract_card(file)
cards_1.append(cards[0])

file = 'train_games/game2/9.jpg'
cards = extract_card(file)
cards_1.append(cards[2])



cards_J =[]
# read "J"s:
file = 'train_games/game1/1.jpg'
cards = extract_card(file)
cards_J.append(cards[2])

file = 'train_games/game1/2.jpg'
cards = extract_card(file)
cards_J.append(cards[1])

file = 'train_games/game1/4.jpg'
cards = extract_card(file)
cards_J.append(cards[1])


file = 'train_games/game1/11.jpg'
cards = extract_card(file)
cards_J.append(cards[2])


file = 'train_games/game2/1.jpg'
cards = extract_card(file)
cards_J.append(cards[0])
cards_J.append(cards[3])


file = 'train_games/game2/2.jpg'
cards = extract_card(file)
cards_J.append(cards[0])


file = 'train_games/game2/6.jpg'
cards = extract_card(file)
cards_J.append(cards[3])



for i in range(len(cards_J)):
        plt.subplot(round(len(cards_J)/4),5,i+1)
        plt.imshow(cards_J[i])
        #cv2.imwrite(f'cards_latest/{file[12:17]}_player{i}_{file[18:]}', cv2.cvtColor(cards[i-1], cv2.COLOR_RGB2BGR))
        cv2.imwrite('J'+ str(i) +'.jpg', cv2.cvtColor(cards_J[i],cv2.COLOR_RGB2BGR) )

plt.show()

for i in range(len(cards_1)):
        plt.subplot(round(len(cards_1)/4),5,i+1)
        plt.imshow(cards_1[i])
        cv2.imwrite('1'+ str(i) +'.jpg', cv2.cvtColor(cards_1[i],cv2.COLOR_RGB2BGR) )


plt.show()

"""

