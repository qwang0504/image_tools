import cv2
from numpy.typing import NDArray
import numpy as np
from .convert import im2gray, im2rgb

def polyroi(img: NDArray) -> NDArray:

    img_rgb = im2rgb(img)
    local_image = img_rgb.copy()
    coords = []

    def click_event(event, x, y, flags, params): 
        nonlocal coords, local_image, img_rgb

        # checking for left mouse clicks 
        if event == cv2.EVENT_LBUTTONDOWN: 
            coords.append((x,y))
    
        # checking for right mouse clicks      
        if event == cv2.EVENT_RBUTTONDOWN: 
            coords.pop()

        pts = np.array(coords, np.int32)
        pts = pts.reshape((-1,1,2))
        
        # it looks like cv2.polylines modifies input inplace
        # so I make a copy
        original = img_rgb.copy() 
        local_image = cv2.polylines(original, [pts], True, (0, 0, 255), 1)
        cv2.imshow('select roi', local_image) 

    cv2.namedWindow('select roi')
    cv2.setMouseCallback('select roi', click_event)
    cv2.imshow('select roi', local_image) 
    cv2.waitKey(0) 
    cv2.destroyWindow('select roi') 
    
    return np.array(coords, np.int32)

def polymask(img: NDArray) -> NDArray:
    mask = np.zeros_like(img)
    coords = polyroi(img)
    mask_RGB = cv2.fillPoly(mask, [coords], 255)
    return im2gray(mask_RGB).astype(np.uint8)

