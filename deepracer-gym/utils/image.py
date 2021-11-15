import gym
import numpy as np
import cv2
import matplotlib.pyplot as plt
import deepracer_gym
from skimage import measure
import numpy as np
import torch
import kornia
import cv2
import matplotlib.pyplot as plt 
import numpy as np

from kornia.morphology import dilation, opening, closing
from kornia.utils import tensor_to_image, draw_rectangle
from kornia.contrib import connected_components
from kornia.geometry.transform import resize

from skimage.filters import threshold_multiotsu

def get_otsu(img, classes=3):
    """Applies otsu multi-class thresholding on an image, by default with 3 classes"""
    thresholds = threshold_multiotsu(img, classes=classes)
    return np.digitize(img, bins=thresholds)*127

def get_disparity(left, right):
    """Applies disparity map from given left and right images"""
    # Pre-processing by mean adjusting the images
    L_gray = left - np.mean(left)
    R_gray = right - np.mean(right)

    # Select block size over here 
    block_size = [11, 11]

    # Call to GPU function
    D_map = host_code.compute_disparity_gpu(L_gray, R_gray, block_size)

    # Smoothening the result by passing it through a median filter
    return cv2.medianBlur(D_map, 13)

def keep_long_components(img, threshold=50):
    """Applies blob detection using skimage measure label and removes all blob with horizontal length less than threshold"""
    blobs_labels = measure.label(img, background=0)
    kept_labels = []

    for lab in range(1, len(np.unique(blobs_labels))):
        x,y = np.nonzero(np.where(blobs_labels == lab, 1, 0))
        
        if np.max(y) - np.min(y) > threshold:
            kept_labels.append(lab)
            
    return np.where(np.isin(blobs_labels, kept_labels), 255, 0)

def segment_img(img, init_thresh=160, denoising=False):
    """Segments road image and returns image with mask of the sidelines only, and middle dotted line only"""
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
    """Applies segment image and resizes image to size given"""
    side, mid = segment_img(img, init_thresh=init_thresh, denoising=denoising)
    return cv2.resize(side+mid, size) > 0

def tensor_to_features(tensor, side_thresh=200, mid_thresh=160, sidelength_threshold=50):
    """Segments tensor into parts of middle dotted line, background and sidelines"""
    batch_size = tensor.shape[0]
    
    # compute thresholded image (only white parts)
    side_thresholded = tensor[:,:,20:] > side_thresh
    mid_thresholded  = tensor[:,:,20:] > mid_thresh
    
    # dilate to keep them together
    dilated = dilation(side_thresholded, torch.ones(5,20, device='cuda:0'), border_type='constant')
    test = dilated
    connected_comp = connected_components(dilated, num_iterations=200)
    
    # keep connected components with large enough horizontal length
    labels = connected_comp.unique(sorted=True)[1:]
    kept_labels = []

    for lab in labels:
        _,_,x,y = torch.nonzero(torch.where(connected_comp == lab, 1, 0), as_tuple=True)

        if torch.max(y) - torch.min(y) > sidelength_threshold:
            kept_labels.append(lab)
            
    side = (connected_comp[..., None] == torch.tensor(kept_labels, device='cuda:0')).any(-1).double()

    test2 = side
    # dilate side line to remove it from rest of image
    dilated_side = dilation(side, torch.ones(10,7,device='cuda:0'), border_type='constant')
    removed_upper = ((mid_thresholded.double() - dilated_side) > 0).double()
    
    # open to remove noise
    open_mid = opening(removed_upper, torch.ones(6,6, device='cuda:0'))
    
    # close to join midlines together
    closed_midline = closing(open_mid, torch.ones(40,40, device='cuda:0'), border_type='geodesic')
    closed_midline = draw_rectangle(closed_midline, torch.tensor([[[0, 0, 160, 120]] for _ in range(batch_size)], device='cuda:0'), fill=torch.zeros(1, device='cuda:0'))
    
    # reassemble together in a single image
    resized_mid  = (resize(closed_midline,(16,16), interpolation='area') > 0.2).double()*0.5
    resized_side = (resize(side,(16,16), interpolation='area') > 0.2).double()
    
    return resized_mid + resized_side