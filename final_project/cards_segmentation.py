import cv2
import numpy as np
from sklearn.cluster import KMeans
from scipy import ndimage
from copy import deepcopy
import os

def mask_range(img):
    
    lower_range = np.array([20,100,20])
    upper_range = np.array([100,200,100])

    mask = cv2.inRange(img, lower_range, upper_range)
    
    return mask

def mask_thre(img):
    
    image = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    image = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    mask = (1-cv2.adaptiveThreshold(image,1,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,101,5))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(10,10))
    mask = ndimage.binary_opening(mask, kernel, iterations=1) 
    
    return mask

def players_clustering(mask):
    seeds_kmean=np.array([[2000,3800],[3000,2200],[2000,800],[1000,2400]])
    tmp = mask
    img = deepcopy(tmp)
    points = []

    minLineLength = 400 #400
    maxLineGap = 150
    lines = cv2.HoughLinesP(img,1,np.pi/180,100,minLineLength=minLineLength,maxLineGap=maxLineGap )
    
    for line in lines:
        for x1,y1,x2,y2 in line:
            points.append([x1,y1])
            points.append([x2,y2])

    points_f=np.array(points)
    kmeans = KMeans(n_clusters=4, init = seeds_kmean, random_state=0).fit(points_f)
    
    labels = kmeans.labels_.ravel()

    players = {new_list: [] for new_list in range(len(np.unique(labels)))}
    for index in range(len(labels)):
        players[labels[index]].append(points_f[index])
        
    centers=[]
    for i in range (4):
        center = np.mean(players[i],axis = 0)
        centers.append(center)

    return players,centers,points_f,kmeans.labels_.ravel()

def find_corners(players,centers,image):
    isolated_cards = []
    left_lowers = []
    right_lowers = []
    right_uppers = []
    for i in range(4):
        player = None
        if centers[i][0] <= 1152 and centers[i][1] >= 1536 and centers[i][1] <= 3072:
            player = 'player4'
        if centers[i][0] >= 1152 and centers[i][0] <= 2304 and centers[i][1] >= 3072:
            player = 'player1'
        if centers[i][0] >= 2304 and centers[i][1] >= 1536 and centers[i][1] <= 3072:
            player = 'player2'
        if centers[i][0] >= 1152 and centers[i][0] <= 2304 and centers[i][1] <= 1536:
            player = 'player3'

        left_lower = [-1,-1]
        left_upper = [-1,-1]
        right_lower = [-1,-1]
        right_upper = [-1,-1]

        dist_max = 0
 
        for coords in players[i]:
            if dist_max<np.sqrt((coords[0]-centers[i][0])**2+(coords[1]-centers[i][1])**2) and (coords[0]<centers[i][0] and coords[1]> centers[i][1]): # left lower
                dist_max = np.sqrt((coords[0]-centers[i][0])**2+(coords[1]-centers[i][1])**2)
                left_lower = coords
                
       
        dist_max = 0
        for coords in players[i]:
            if dist_max<np.sqrt((coords[0]-centers[i][0])**2+(coords[1]-centers[i][1])**2) and (coords[0]<centers[i][0] and coords[1]< centers[i][1]): # left upper
                dist_max = np.sqrt((coords[0]-centers[i][0])**2+(coords[1]-centers[i][1])**2)
                left_upper = coords
                

        dist_max = 0
        for coords in players[i]:
            if dist_max < np.sqrt((coords[0]-centers[i][0])**2+(coords[1]-centers[i][1])**2) and (coords[0]>centers[i][0] and coords[1]>centers[i][1]): # right lower
                
                dist_max = np.sqrt((coords[0]-centers[i][0])**2+(coords[1]-centers[i][1])**2)
                right_lower = coords
                
                
        dist_max = 0
        for coords in players[i]:
            if dist_max<np.sqrt((coords[0]-centers[i][0])**2+(coords[1]-centers[i][1])**2) and (coords[0]>centers[i][0] and coords[1]<centers[i][1]): # right upper
                dist_max = np.sqrt((coords[0]-centers[i][0])**2+(coords[1]-centers[i][1])**2)
                right_upper = coords
        
        if right_lower[0] == -1:
            right_lower[0] =  2 * int((right_upper[0] + left_lower[0]) / 2) - left_upper[0]
            right_lower[1] =  2 * int((right_upper[1] + left_lower[1]) / 2) - left_upper[1]
        
        if left_lower[0] == -1:
            left_lower[0] =  2 * int((left_upper[0] + right_lower[0]) / 2) - right_upper[0]
            left_lower[1] =  2 * int((left_upper[1] + right_lower[1]) / 2) - right_upper[1]
            
        if right_upper[0] == -1:
            right_upper[0] =  2 * int((left_upper[0] + right_lower[0]) / 2) - left_lower[0]
            right_upper[1] =  2 * int((left_upper[1] + right_lower[1]) / 2) - left_lower[1]
        

        left_lowers.append(left_lower)
        right_lowers.append(right_lower)
        right_uppers.append(right_upper)

        isolated = affin_transform(player,left_lower,right_lower,right_upper,image)
        isolated_cards.append(isolated)

    return isolated_cards,left_lowers,right_lowers,right_uppers

def affin_transform(player,left_lower,right_lower,right_upper,image):
    input_pts = np.float32([left_lower,right_lower,right_upper])
    cols=300
    rows=400
    if player == 'player1':
        output_pts = np.float32([[0,rows],[cols,rows],[cols,0]])
    if player == 'player2':
        output_pts = np.float32([[0,0],[0,rows],[cols,rows]])
    if player == 'player3':
        output_pts = np.float32([[cols,0],[0,0],[0,rows]])
    if player == 'player4':
        output_pts = np.float32([[cols,rows],[cols,0],[0,0]])

    M= cv2.getAffineTransform(input_pts , output_pts)
    res = cv2.warpAffine(image, M, (cols,rows))
    return res

def walkFile(file):
    file_list =[]
    for root, dirs, files in os.walk(file):
        for f in files:

            #print(os.path.join(root, f))
            file_list.append(os.path.join(root, f))

    return file_list

def read_image(filename):
    img = cv2.imread(filename)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img