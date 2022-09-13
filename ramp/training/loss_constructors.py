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

import tensorflow as tf
import keras.backend as K
from tensorflow.keras.losses import CategoricalCrossentropy, categorical_crossentropy

import numpy as np


def get_sparse_categorical_crossentropy_fn(cfg):
    '''
    returns sparse categorical cross entropy. Does not use parameters. 
    '''
    return tf.keras.losses.sparse_categorical_crossentropy

def get_custom_loss_fn(cfg):    
    '''
    returns the Google Africa custom loss function with given parameters.
    sigma: length scale for the gaussian-filtered weight image in the weighted scce loss. 
    myconst: constant multiple for the gaussian-filtered weight image in the weighted scce loss.
    gamma (real, >0): FTL power parameter.
    alpha (real, 0-1): trades off cost of FP vs FN. alpha larger => FP counts more.
    epsilon (real,>0): small positive number to stabilize FTL computations.
    ftl_wt (real): trades off FT loss vs wtd scce loss. flt_wt larger => FT loss counts more.
    '''           
    
    ftl_wt = cfg["loss"]["loss_fn_parms"]["ftl_wt"]

    def custom_loss_fn(y_true, y_hat):
        wscce_loss = get_weighted_SCCE_loss_fn(cfg)
        ft_loss = get_focal_tversky_loss_function(cfg)
        # return wscce_loss(y_true, y_hat) + 
        return ftl_wt*ft_loss(y_true, y_hat)
    return custom_loss_fn



def get_weighted_SCCE_loss_fn(cfg):
    '''
    Returns a sparse categorical cross entropy loss function with per-pixel weighting.
    '''
    from ..utils.imgproc_utils import batch_get_weight_image
    num_classes = cfg["num_classes"]
    sigma = cfg["loss"]["loss_fn_parms"]["sigma"]
    myconst = cfg["loss"]["loss_fn_parms"]["myconst"]
    assert num_classes == 2, "Multiclass (>2) masks are not currently supported"

    def weighted_SCCE_loss(y_true, y_hat):

        xentropy_loss = CategoricalCrossentropy()

        # get the weighting image from the mask. 
        # it has shape: (B,H,W)
        wt_img = batch_get_weight_image(y_true, sigma, myconst)

        # one hot encode and squeeze y_true to have shape (B, H, W, num_classes)
        yt = tf.squeeze(tf.one_hot(tf.cast(y_true, tf.int32), num_classes))
        return xentropy_loss(yt, y_hat, sample_weight=wt_img)

    return weighted_SCCE_loss



def get_SCCE_labelsmoothing_loss_fn(cfg):
    #num_classes, ls_param=0.1):
    '''
    Returns a sparse categorical cross entropy loss function that applies label smoothing.
    '''
    num_classes = cfg["num_classes"]
    ls_param = cfg["loss"]["loss_fn_parms"]["ls_param"]
    def SCCE_labelsmoothing(y_true, y_hat):
        yt = tf.squeeze(tf.one_hot(tf.cast(y_true, tf.int32), num_classes))
        return categorical_crossentropy(yt, y_hat, label_smoothing = ls_param)

    return SCCE_labelsmoothing


def get_class_tversky_fn(alpha, epsilon):
    
    def class_tversky(y_true, y_pred):
        '''
        Returns a vector with length equal to the number of classes.
        Each class contributes to the total FTL in a 1-vs-rest manner.
        '''

        # Note: batch_flatten actually flattens all the other dimensions. 
        # So, for example, a (3, 5, 6, 2)-tensor becomes a (3, 60)-tensor.
        # In this case, because you permuted channels and batches, you 
        # get a (C, HWB)-tensor.
        y_true_pos = K.batch_flatten(y_true)
        y_pred_pos = K.batch_flatten(y_pred)

        # Note that each of these sums is over the HWB dimension, so you 
        # get a C-vector in each case. 
        # TP(c): terms are 0 if true class isn't c, large if the true class was 
        # c and you got it right.
        true_pos = K.sum(y_true_pos * y_pred_pos, 1)
        
        # FN(c): terms are 0 if true class isn't c, large if the true class was 
        # c and you got it wrong.
        false_neg = K.sum(y_true_pos * (1-y_pred_pos), 1)
        
        # FP(c): terms are 0 if true class is c. and large if the true class 
        # wasn't c, but you guessed that it was.
        false_pos = K.sum((1-y_true_pos)*y_pred_pos, 1)
        
        # alpha trades off FP and FN.
        # epsilon stabilizes the division.
        return (true_pos + epsilon)/(true_pos + alpha*false_neg + (1-alpha)*false_pos + epsilon)

    return class_tversky

def get_focal_tversky_loss_function(cfg):
    #num_classes, gamma, alpha, epsilon):
    '''
    returns a FTL function with the given parameters.
    num_classes (integer): # classes in the mask. 
    gamma (real): FTL main parameter.
    alpha (real): trades off cost of FP vs FN.
    epsilon (real): small factor to stabilize FTL computations.
    '''
    
    num_classes = cfg["num_classes"]
    gamma = cfg["loss"]["loss_fn_parms"]["gamma"]
    alpha = cfg["loss"]["loss_fn_parms"]["alpha"]
    epsilon = cfg["loss"]["loss_fn_parms"]["epsilon"]

    class_tversky = get_class_tversky_function(alpha, epsilon)

    def focal_tversky_loss(y_true, y_pred):
        '''
        returns the sum of the classwise losses, raised to the gamma
        power to make it nonlinear.
        '''
        # y_true and y_pred must be one-hot encoded. 
        # usually, y_true is sparse-encoded, and y_pred is one-hot encoded. 
        # so transform y_true first.
        yt = tf.one_hot(tf.cast(tf.squeeze(y_true, 3), tf.int32), num_classes)

        # for tensors of shape batch-H-W-channels, this trades
        # the batch and channels dims, so you get a channels-H-W-batch shape.
        yt = K.permute_dimensions(yt, (3,1,2,0))
        y_pred = K.permute_dimensions(y_pred, (3,1,2,0))
        
        # calculate classwise components of FTL
        pt_1 = class_tversky(yt, y_pred)
        return K.sum(K.pow((1-pt_1), gamma))

    return focal_tversky_loss


def main():

    # unit test
    num_classes = 3
    y_true = np.array((0,1,2,0,1,2,0,1,2))

    # this guesses 0 a lot, and 2, hardly ever. 
    y_pred_0 = np.array((0.25, 0.25, 0.5, 0.5, 0.5, 0.5, 0.5, 0.25, 0.25))
    y_pred_1 = np.array((0.5, 0.5, 0.25, 0.25, 0.25, 0.25, 0.25, 0.5, 0.25))
    y_pred_2 = np.array((0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.5))

    # shape is (9, 3) -- classes last as comes out of the NN
    y_pred = np.array((y_pred_0, y_pred_1, y_pred_2)).astype('float32').transpose()
    

    # one hot encode the truth vector
    yt = tf.squeeze(tf.one_hot(tf.cast(y_true, tf.int32), num_classes))
    
    yt = K.permute_dimensions(yt, (1,0))
    yp = K.permute_dimensions(y_pred, (1,0))

    class_tversky = get_class_tversky_function(alpha=0.5, epsilon=0)
    CT = class_tversky(yt, yp)
    truth = np.array((0.38462, 0.41667,0.36364))
    assert np.sum(np.square(CT - truth)) < 10E-06, f"returned unexpected value from class_tversky(): {CT}"

    FT_loss = get_focal_tversky_loss_function(3, 1.0, 0.5, 0.0)
    # reshape y_true and y_pred so they look like tf tensors

    y_true = y_true.reshape((1, 3, 3))
    y_pred = y_pred.reshape((1, 3, 3, 3))

    FTL = FT_loss(y_true, y_pred)
    truth_ftl = 1.83507
    print(FTL)
    assert np.square(FTL - truth_ftl) < 10E-06, f"returned unexpected FTL: {FTL}"
    

    return


    





if __name__=='__main__':
    main()
