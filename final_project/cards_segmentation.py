import cv2
import numpy as np
from sklearn.cluster import KMeans
from scipy import ndimage
from copy import deepcopy
import os

def mask_range(img):
    """
    Apply a green color mask to the image (without dealer), in order to find the cards' edges.
    Args: 
        img: the image with 4 players (without dealer)
    Return: 
        mask: the binary image where only the pixels which are in the green range are 1, all
              the others pixels' intensity are 0
    """
    
    lower_range = np.array([20,100,20])
    upper_range = np.array([100,200,100])

    mask = cv2.inRange(img, lower_range, upper_range)
    
    return mask

def mask_thre(img):
    """
    Apply an adaptive threshold to the image (with dealer), in order to find all edges.
    Args: 
        img: the image with 4 players and dealer
    Return: 
        mask: the binary image after the adaptive threshold
    """
    
    image = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    mask = (1-cv2.adaptiveThreshold(image,1,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,101,5))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(10,10))
    mask = ndimage.binary_opening(mask, kernel, iterations=1) 
    
    return mask

def players_clustering(mask):
    """
    Perform an Hough line detection to find the points on the mask, then use KMeans algorithm to cluster these points 
    to 4 players according to their coordinates. Once the coordinates of all players are found, we compute the center 
    points for each players.
    Args: 
        mask: the binary image with 4 players(without dealer)
    Return: 
        players: dictionary which contains the players(1,2,3,4) and the points on the edges which belongs to each player
        centers: the center of all coordinates which belongs to each player
        points_f: a list which contains all the points found with Hough line detection method
        kmeans.labels_.ravel(): the list of the labels by players, corresponds to points_f
    """
    seeds_kmean=np.array([[2000,3800],[3000,2200],[2000,800],[1000,2400]])
    tmp = mask
    img = deepcopy(tmp)
    points = []

    minLineLength = 400 
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
    """
    This fonction calculates four corners of each card for each player
    Args: 
        players: dictionary which contains the players(1,2,3,4) and the points on the edges which belongs to each player
        centers: the center of all coordinates which belongs to each player
        image: the image with 4 players (without dealer), it will be used to crop the card of each player 
    Return: 
        isolated_cards: the list containing 4 resulting cards, cropped and rotated
        left_lowers: a list containing the left lower corners for each players
        right_lowers: a list containing the right lower corners for each players
        right_uppers: a list containing the right corners for each players
    """
    
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
    """
    This fonction receives 3 corners' corrdinates of each card for each player, then it apply the affin transform to rotate
    card in the good orientation, according to which player the corners belongs to. 
    Args: 
        player: a string indicating which player it is currently looking at
        left_lower: lower left coordinate of the card 
        right_lower: lower right coordinate of the card 
        right_upper: upper right coordinate of the card 
        image: the image with 4 players (without dealer), it will be used to crop the card of each player 
    Return: 
        res:the resulting card of this current player, cropped and rotated
    """
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
    """
    This founction return a list of files by giving a path 
    Args: 
        file: the file path in which we want to visit all the files in this file (or files in its subfiles)
    Return: 
        file_list : a list containing all the files in the giving file path
    """
    file_list =[]
    for root, dirs, files in os.walk(file):
        for f in files:

            #print(os.path.join(root, f))
            file_list.append(os.path.join(root, f))

    return file_list

def read_image(filename):
    """
    This founction return a color-converted(from BGR to RGB) image 
    Args: 
        filename: the file name of image we want to read
    Return: 
        img : the image after the conversion from BGR to RGB
    """
    img = cv2.imread(filename)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img