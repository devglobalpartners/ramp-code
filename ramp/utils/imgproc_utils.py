#################################################################
#
# created for ramp project, August 2022
# Author: carolyn.johnston@dev.global
#
#################################################################



# adding logging
import logging
log = logging.getLogger(__name__)
log.addHandler(logging.NullHandler())

import numpy as np
import cv2
import tensorflow as tf

def get_edge_image(mask_tensor):
    '''
    takes a mask tensor with shape (H,W) or (H,W,1)
    returns a (H,W)-shaped float32 numpy array with edge points marked as 1.
    '''
    # mask tensor will frequently have shape (H,W,1)
    # we need (H,W)
    mask = np.squeeze(mask_tensor.numpy())

    # even though the edge_image is 0,1-valued,
    # it has to be a float32 or gaussian blurring will break. 
    edge_image = np.zeros(mask.shape, dtype=np.float32 )

    # contours is a dataset of boundary points
    # CHAIN_APPROX_NONE means all points are retained 
    # for burning into the edge image 
    contours, t2 = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    for contour in contours:
        
        # contour has dimensions: [#pts in contour, 1, 2]
        # get rid of the extra 1...
        pts_array = np.squeeze(contour, 1)

        # turn pts_array of shape [#pts, 2] into a list of x,y pairs
        pts = [pts_array[ii,:] for ii in range(pts_array.shape[0])]

        # mark every border point in the edge image.
        for pt in pts:
            edge_image[pt[1], pt[0]] = 1
    

    return edge_image

def get_weight_image(mask_tensor, sigma, myconst):
    '''
    takes uint8 binary mask tensor of dimensions (H,W) or (H,W,1)
    sigma is scale length for gaussian kernel in pixels
    myconst is a multiplicative constant applied to the weight image
    returns float64 weight image of dimensions (H,W)
    '''
    edge_image = get_edge_image(mask_tensor)
    
    # important undocumented note: the blur image will have the same datatype as the edge image. 
    # so if the edge image is uint8, the blur image will be 0. 
    blur_image = cv2.GaussianBlur(edge_image, 
                                    ksize=(0,0), 
                                    sigmaX=sigma, 
                                    sigmaY=sigma, 
                                    borderType=cv2.BORDER_CONSTANT)

    return myconst*blur_image

def batch_get_weight_image(mask_tensor_batch, sigma, myconst):
    '''
    runs get_weight_image over a batch of mask tensors.
    batch dimension is first.
    returns numpy array of shape (B, H, W).
    '''
    
    # this will be (B,H,W,1) for sparse-encoded masks
    (B, H, W, C) = mask_tensor_batch.shape
    batch_weight_img = np.zeros([B, H, W])

    for ii in range(B):
        mask_tensor_ii = mask_tensor_batch[ii,...]
        batch_weight_img[ii,...] = get_weight_image(mask_tensor_ii, sigma, myconst)

    return batch_weight_img
