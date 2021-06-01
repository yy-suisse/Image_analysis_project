import cv2
import numpy as np

def mask_dealer(img):
    """
    This function allows us to find the position of dealer, it will also apply a mask to remove all the pixels in the dealer area
    Args: 
        img: the RGB image with 4 players and dealer
    Return: 
        img: the RGB image with 4 players, all the pixels belonging to the dealer are removed 
        center: the coodinate of dealer's center 
    """
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
            center = (i[0], i[1])
            cv2.circle(img, center, i[2]+50, (255,255,255), thickness = -1)
            
    return img, center