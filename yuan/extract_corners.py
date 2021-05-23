import cv2
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from sklearn.cluster import KMeans



images = []
#plt.figure(figsize=(12, 24)) 

for i in range(1):
    for j in range(13):
        img = cv2.imread(f"train_games/game{i+1}/{j+1}.jpg")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        images.append(img)
        #plt.subplot(5,3,j+1)
        #plt.imshow(img)


################################################################################### find dealer

images_C = []
#plt.figure(figsize=(12, 24)) 

count = 0
for img in images:
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    blurred = cv2.medianBlur(gray, 25) #cv2.bilateralFilter(gray,10,50,50)

    minDist = 100
    param1 = 30 #500
    param2 = 50 #200 #smaller value-> more false circles
    minRadius = 300
    maxRadius = 500 #10

    # docstring of HoughCircles: HoughCircles(image, method, dp, minDist[, circles[, param1[, param2[, minRadius[, maxRadius]]]]]) -> circles
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, 1, minDist, param1=param1, param2=param2, minRadius=minRadius, maxRadius=maxRadius)

    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0,:]:
            cv2.circle(img, (i[0], i[1]), i[2], (255, 0, 0), thickness = 10)
            cv2.circle(img, (i[0], i[1]), i[2], (255, 0, 0), thickness = -1)
    count += 1
    images_C.append(img)
    # plt.subplot(5,3,count)
    # plt.imshow(img)

################################################################################### add mask


masks = []
#plt.figure(figsize=(12, 24)) 

count = 0

for img in images_C:

    lower_range = np.array([20,100,20])
    upper_range = np.array([100,200,100])

    mask = cv2.inRange(img, lower_range, upper_range)
    
    masks.append(mask)

    count += 1
    #plt.subplot(5,3,count)
    #plt.imshow(mask)

################################################################################### kmean to find points for all players for one picture (first one)

#plt.figure(figsize=(12, 24)) 
#count = 0
seeds_kmean=np.array([[1000,2400],[2000,800],[2000,3800],[3000,2200]])
tmp = masks[0]
img = deepcopy(tmp)
img2 = np.zeros_like(tmp)
points = []

minLineLength = 400
maxLineGap = 150
lines = cv2.HoughLinesP(img,1,np.pi/180,100,minLineLength=minLineLength,maxLineGap=maxLineGap )
print(np.shape(lines))
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
kmeans = KMeans(n_clusters=4, random_state=0).fit(points_f)
#plt.imshow(img2)
#plt.scatter(points_f[:, 0], points_f[:, 1], c=kmeans.labels_.ravel(), s=10, lw=0, cmap='RdYlGn')


################################################################################### find points for different playsers (for picture 1)


labels = kmeans.labels_.ravel()

players = {new_list: [] for new_list in range(len(np.unique(labels)))}
#print(players)
for index in range(len(labels)):
    players[labels[index]].append(points_f[index])
    
centers=[]
for i in range (4):
    #print(np.mean(players[i],axis = 0))
    #print(players[i])
    center = np.mean(players[i],axis = 0)
    #print(center)
    centers.append(np.mean(players[i],axis = 0))
    #print(centers)
    plt.imshow(masks[0])
    plt.scatter(points_f[:, 0], points_f[:, 1], c=kmeans.labels_.ravel(), s=10, lw=0, cmap='RdYlGn')
    plt.scatter(center[0], center[1],s=10, lw=0, cmap='RdYlGn')
    
#plt.show()

################################################################################### find corners for different playsers (for picture 1)
x_min = 5000
y_min = 5000
x_max= 0
y_max = 0
left_lower = []
left_upper = []
right_lower = []
right_upper = []
for coords in players[0]:
    if coords[1]>y_max and (coords[0]<centers[0][0] and coords[1]> centers[0][1]): # left lower
        y_max = coords[1]
        left_lower = coords
        
x_min = 5000
y_min = 5000
x_max= 0
y_max = 0

for coords in players[0]:
    if coords[1]<y_min and (coords[0]<centers[0][0] and coords[1]< centers[0][1]): # left upper
        y_min = coords[1]
        left_upper = coords
        

x_min = 5000
y_min = 5000
x_max= 0
y_max = 0

for coords in players[0]:
    if coords[1]>y_max and (coords[0]>centers[0][0] and coords[1]>centers[0][1]): # right lower
        
        y_min = coords[1]
        right_lower = coords
        
        
x_min = 5000
y_min = 5000
x_max= 0
y_max = 0

for coords in players[0]:
    if coords[1]<y_min and (coords[0]>centers[0][0] and coords[1]<centers[0][1]): # right upper
        y_max = coords[1]
        right_upper = coords
        
        
print("left lower:",left_lower)
print("left upper:",left_upper)
print("right lower:",right_lower)
print("right upper:",right_upper)

################################################################################### affine transform


plt.figure(figsize=(12, 24)) 
#count = 0
#player = [player1, player2, player3, player4]
input_pts = np.float32([left_upper,right_lower,right_upper])
cols=300
rows=400
cols=300
rows=400

output_pts = np.float32([[0,0],[cols,rows],[cols,0]])

M= cv2.getAffineTransform(input_pts , output_pts)

res = cv2.warpAffine(images[0], M, (cols,rows))

plt.subplot(1,2,1)
plt.imshow(res)
plt.subplot(1,2,2)
plt.imshow(images[0])


plt.show()

