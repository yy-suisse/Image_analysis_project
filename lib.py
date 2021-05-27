import cv2
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from sklearn.cluster import KMeans
import os
from scipy import ndimage

def read_image(filename):
    img = cv2.imread(filename)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img



def mask_dealer(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    blurred = cv2.medianBlur(gray, 25) #cv2.bilateralFilter(gray,10,50,50)

    minDist = 100
    param1 = 30 #500
    param2 = 50 #200 #smaller value-> more false circles
    minRadius = 250
    maxRadius = 500 #10

    # docstring of HoughCircles: HoughCircles(image, method, dp, minDist[, circles[, param1[, param2[, minRadius[, maxRadius]]]]]) -> circles
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, 1, minDist, param1=param1, param2=param2, minRadius=minRadius, maxRadius=maxRadius)

    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0,:]:
            cv2.circle(img, (i[0], i[1]), i[2], (255, 0, 0), thickness = 10)
            cv2.circle(img, (i[0], i[1]), i[2], (255, 0, 0), thickness = -1)
    return img


def mask_for_extract_cards(img):
    lower_range = np.array([20,100,20])
    upper_range = np.array([100,200,100])

    mask = cv2.inRange(img, lower_range, upper_range)
    
    #kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    
    #mask = ndimage.binary_opening(mask, kernel, iterations=1) 
    
    return mask



def players_clustering(mask):
    seeds_kmean=np.array([[2000,3800],[3000,2200],[2000,800],[1000,2400]])
    tmp = mask
    img = deepcopy(tmp)
    img2 = np.zeros_like(tmp)
    points = []

    minLineLength = 400 #400
    maxLineGap = 150
    lines = cv2.HoughLinesP(img,1,np.pi/180,100,minLineLength=minLineLength,maxLineGap=maxLineGap )
    #print(np.shape(lines))
    for line in lines:
        for x1,y1,x2,y2 in line:
            #print(x1,y1,x2,y2)
            #cv2.line(img2, ( x1,y1 ),( x2,y2 ),( 255,255,255 ),2 )
            points.append([x1,y1])
            points.append([x2,y2])
            #cv2.circle(img2, (x1,y1), radius=30, color=(255, 255, 255), thickness=-1)
            #cv2.circle(img2, (x2,y2), radius=30, color=(255, 255, 255), thickness=-1)
    #count += 1
    #plt.subplot(5,3,count)     
    #plt.subplot(1,1,1)     

    points_f=np.array(points)
    kmeans = KMeans(n_clusters=4, init = seeds_kmean, random_state=0).fit(points_f)
    

    labels = kmeans.labels_.ravel()

    players = {new_list: [] for new_list in range(len(np.unique(labels)))}
    #print(players)
    for index in range(len(labels)):
        players[labels[index]].append(points_f[index])
        
    centers=[]
    for i in range (4):
        center = np.mean(players[i],axis = 0)
        #print(center)
        centers.append(np.mean(players[i],axis = 0))
        #plt.imshow(mask)
        #plt.scatter(points_f[:, 0], points_f[:, 1], c=kmeans.labels_.ravel(), s=10, lw=0, cmap='RdYlGn')
        #plt.scatter(center[0], center[1],s=10, lw=0, cmap='RdYlGn')
        #plt.show()
    return players,centers,points_f,kmeans.labels_.ravel()

def find_corners(players,centers,image):
    isolated_cards = []
    left_lowers = []
    #left_uppers = []
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
        
        print("player:",player)

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
        #left_uppers.append()
        right_lowers.append(right_lower)
        right_uppers.append(right_upper)
        
        print("left lower:",left_lower)
        print("left upper:",left_upper)
        print("right lower:",right_lower)
        print("right upper:",right_upper)
        
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



def white_balance_1(img):
    '''
    第一种简单的求均值白平衡法
    :param img: cv2.imread读取的图片数据
    :return: 返回的白平衡结果图片数据
    '''
    # 读取图像
    r, g, b = cv2.split(img)
    r_avg = cv2.mean(r)[0]
    g_avg = cv2.mean(g)[0]
    b_avg = cv2.mean(b)[0]
    # 求各个通道所占增益
    k = (r_avg + g_avg + b_avg) / 3
    kr = k / r_avg
    kg = k / g_avg
    kb = k / b_avg
    r = cv2.addWeighted(src1=r, alpha=kr, src2=0, beta=0, gamma=0)
    g = cv2.addWeighted(src1=g, alpha=kg, src2=0, beta=0, gamma=0)
    b = cv2.addWeighted(src1=b, alpha=kb, src2=0, beta=0, gamma=0)
    balance_img = cv2.merge([r, g, b])
    return balance_img


#####

file_list = walkFile("train_games")
file_pics=[]
print(file_list)
for file in file_list:
    if file.endswith('.jpg'):
       file_pics.append(file)

print(file_pics)

file_pics = ['train_games\\game7\\3.jpg']

for file in file_pics:
    #print(file[18:])
    '''
    if file == 'train_games\game4\8.jpg':
        continue
    
    if file == 'train_games\game5\8.jpg':
        continue
    
    if file == "train_games\game6\13.jpg":
        print('hello!')
        continue
    '''
    
    image = read_image(file)
    
    #image = white_balance_1(image)
    
    image_dealer = mask_dealer(image)
    #plt.imshow(image_dealer)

    extracted = mask_for_extract_cards(image_dealer)
    plt.imshow(extracted)
    '''
    players,centers,points_f,labels = players_clustering(extracted)
    cards,left_lowers,right_lowers,right_uppers = find_corners(players,centers,image)

# plots for debug
    c_x=[]
    c_y=[]
    for center in centers:
        c_x.append(center[0])
        c_y.append(center[1])

    corner_x = []
    corner_y = []
    for i in range(4):
        corner_x.append(left_lowers[i][0])
        corner_x.append(right_lowers[i][0])
        corner_x.append(right_uppers[i][0])
        corner_y.append(left_lowers[i][1])
        corner_y.append(right_lowers[i][1])
        corner_y.append(right_uppers[i][1])
        



# resulting plots
    for i in range(1,5):
        plt.subplot(2,3,i)
        plt.imshow(cards[i-1])
        cv2.imwrite(f'cards/{file[12:17]}_player{i}_{file[18:]}', cv2.cvtColor(cards[i-1], cv2.COLOR_RGB2BGR))  

    plt.subplot(2,3,5)
    plt.scatter(c_x, c_y,s=10, lw=0, cmap='RdYlGn')
    plt.imshow(extracted)

    plt.subplot(2,3,6)
    plt.scatter(points_f[:, 0], points_f[:, 1], c=labels, s=10, lw=0, cmap='RdYlGn')
    plt.scatter(corner_x, corner_y,s=10, lw=0, cmap='RdYlGn')
    plt.imshow(extracted)
    plt.show()
'''