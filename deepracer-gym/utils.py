import gym
import numpy as np
import cv2
import matplotlib.pyplot as plt
import deepracer_gym
from skimage import measure
import numpy as np

def keep_long_components(img, threshold=50):
    
    blobs_labels = measure.label(img, background=0)
    kept_labels = []

    for lab in range(1, len(np.unique(blobs_labels))):
        x,y = np.nonzero(np.where(blobs_labels == lab, 1, 0))
        
        if np.max(y) - np.min(y) > threshold:
            kept_labels.append(lab)
            
    return np.where(np.isin(blobs_labels, kept_labels), 255, 0)

def segment_img(img, init_thresh=160, denoising=False):

    if denoising:
        img = cv2.fastNlMeansDenoising(img.astype('uint8'),20,10,21)
        
    thresholded = (img > init_thresh).astype('uint8')*255
    
    # dilation kernel to select only rotating road on side
    kernel = np.ones((5,5),np.uint8)
    dilation = cv2.dilate(thresholded,kernel,iterations = 1)
    long_comp = keep_long_components(dilation)
    
    # removing rest to keep only middle road
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, ksize=(60,7))
    dilation3 = cv2.dilate(long_comp.astype('uint8'),kernel,iterations = 1, borderType = cv2.BORDER_REFLECT)
    center = (((thresholded - dilation3) > 1).astype('float'))
    
    # add borders so opening doesn't mess up on corners
    outputImage = cv2.copyMakeBorder(
                center, 
                 50, 
                 50, 
                 50, 
                 50, 
                 cv2.BORDER_CONSTANT, 
                 value=0
              )
    
    # apply closing
    kernel = np.ones((50,50),np.uint8)
    middle_road = cv2.morphologyEx(outputImage, cv2.MORPH_CLOSE, kernel)[50:-50, 50:-50]
    
    return long_comp.astype(float)/255, middle_road

def segment_resize(img, init_thresh=160, denoising=False, size=(16,16)):
    side, mid = segment_img(img, init_thresh=init_thresh, denoising=denoising)
    return cv2.resize(side+mid, size) > 0