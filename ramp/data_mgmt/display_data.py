#################################################################
#
# created for ramp project, August 2022
# Author: carolyn.johnston@dev.global
#
#################################################################


import os
import io
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.image import decode_png
from tensorflow import expand_dims
import tensorflow.summary as tf_summary

from ..utils.img_utils import gdal_get_image_tensor, gdal_get_mask_tensor
from ..utils.file_utils import get_basename
from .data_generator import test_batches_from_gtiff_dirs

# adding logging
import logging
log = logging.getLogger(__name__)
log.addHandler(logging.NullHandler())


# Taken from a tensorboard image logging demo at: https://www.tensorflow.org/tensorboard/image_summaries
def plot_to_image(figure):
  """
  Converts the matplotlib plot specified by 'figure' to a PNG image and
  returns it. 
  The supplied figure is closed and inaccessible after this call.
  This code is from a tensorboard/tensorflow image logging demo at: https://www.tensorflow.org/tensorboard/image_summaries
  """
  # Save the plot to a PNG in memory.
  buf = io.BytesIO()
  plt.savefig(buf, format='png')
  # Closing the figure prevents it from being displayed directly inside
  # the notebook.
  plt.close(figure)
  buf.seek(0)
  # Convert PNG buffer to an RGBA (RGB - alpha) TF image to be logged
  image = decode_png(buf.getvalue(), channels=4)
  # Add the batch dimension
  image = expand_dims(image, 0)
  return image


def get_mask_from_prediction(prediction):
    """
    Quick utility to display a model's prediction on a test batch.
    """
    predmask = np.argmax(prediction, axis=-1)
    predmask = np.expand_dims(predmask, axis=-1)
    return predmask

def get_offset_mask_from_prediction(prediction, mask_offset=0):
    """
    Utility to obtain multichannel or binary mask from a one-hot-encoded prediction
    """
    predmask = np.argmax(prediction, axis=-1)
    predmask = np.expand_dims(predmask, axis=-1)
    return predmask + mask_offset

###########################################################################################

def display_a_batch(a_batch):
    '''
    a_batch: a 2-tuple of the form (image_batch, mask_batch)
    image_batch has shape: (batch_size, height, width, 3)
    mask_batch has shape: (batch_size, height, width, 1)
    displays image data in a jupyter notebook 
    '''

    # zip the data together as:
    # (image-mask_1|image-mask_2|...|image-mask_B)
    # where B is batch size
    datapts = list(zip(a_batch[0], a_batch[1]))
    B = len(datapts)

    plt.figure(figsize=(15,(0.5*B*15)))
    
    title = ['Image', 'Mask']
    
    for ii, dd in enumerate(datapts):        
        for jj in range(2):
            plt.subplot(B, 2, ii*2 + jj +1)
            plt.title(title[jj])
            plt.imshow(dd[jj])
    plt.axis('off')
    plt.show()

    ###########################################################################################

# added 20220713

def get_img_mask_pred_batch_figure_with_id(b_idstr, b_img, b_mask, b_predmask):
    '''
    b_idstr: batch id string (list of chips)
    image batch has shape: (batch_size, height, width, 3)
    mask batch and predmask batch both have shape: (batch_size, height, width, 1).
    Run 'get_mask_from_prediction' on the output of model.predict to get b_predmask before calling this function.
    It displays image data in a jupyter notebook.
    '''

    # extract the tile ids from the list of filepaths.
    tile_ids = [get_basename(fpath) for fpath in b_idstr]

    # zip the data together as:
    # (id-image-mask-pred_1|id-image-mask-pred_2|...|id-image-mask-pred_B)
    # where B is batch size
    datapts = list(zip(tile_ids, b_img, b_mask, b_predmask))
    B = len(datapts) # batch size

    figure = plt.figure(figsize=(15,(0.5*B*15)))

    

    for ii, dd in enumerate(datapts):

        # implant chip name in title
        title = [f'Image ({dd[0]})', 'Mask', "Predicted"]
        for jj in range(1,4):
            plt.subplot(B, 3, ii*3 + jj)
            plt.title(title[jj-1])
            plt.imshow(dd[jj])
    plt.axis('off')
    return figure

###################################################################    

def get_img_mask_pred_batch_figure(b_img, b_mask, b_predmask):
    '''
    image batch has shape: (batch_size, height, width, 3)
    mask batch and predmask batch both have shape: (batch_size, height, width, 1).
    Run 'get_mask_from_prediction' on the output of model.predict to get b_predmask before calling this function.
    It displays image data in a jupyter notebook.
    '''
    # zip the data together as:
    # (image-mask-pred_1|image-mask-pred_2|...|image-mask-pred_B)
    # where B is batch size
    datapts = list(zip(b_img, b_mask, b_predmask))
    B = len(datapts) # batch size

    figure = plt.figure(figsize=(15,(0.5*B*15)))

    title = ['Image', 'Mask', "Predicted"]

    for ii, dd in enumerate(datapts):
        for jj in range(3):
            plt.subplot(B, 3, ii*3 + jj +1)
            plt.title(title[jj])
            plt.imshow(dd[jj])
    plt.axis('off')
    return figure


def display_img_mask_pred_batch(b_img, b_mask, b_predmask):
    figure = get_img_mask_pred_batch_figure(b_img, b_mask, b_predmask)
    plt.show()


def display_image_mask(image_path, mask_path):
    '''
    Display the first image passed as an RGB image, 
    and the second as a grayscale mask.
    The user's in charge of making sure they match.
    '''

    img_tensor = gdal_get_image_tensor(image_path)
    mask_tensor = gdal_get_mask_tensor(mask_path)
    trivial_batch = (img_tensor[np.newaxis, :,:,:], mask_tensor[np.newaxis, :,:,:])
    display_a_batch(trivial_batch)
    return

def get_prediction_logging_function(log_dir, model, test_img_dir, test_mask_dir, batch_size, input_image_size, output_image_size):
    '''
    Defines and return a callback function that will log predicted segmentation images 
    from a fixed test data batch after every epoch, during training.
    :param str log_dir: directory to log images to (in the /predictions folder)
    :param str model: model being fitted
    :param str test_img_dir: path to directory with test images
    :param str test_mask_dir: path to directory with test masks
    :param int batch_size: number of images from the beginning of the test image list to use in logging
    :param tuple input_image_size: 2-tuple of integers (width and height of model input images and masks)
    :param tuple output_image_size: 2-tuple of integers (width and height of model output images and masks)
    '''
    log.debug(f"logging predicted masks to logging directory: {log_dir}")
    log.debug(f"model class: {type(model)}")
    log.debug(f"test image directory: {test_img_dir}")
    log.debug(f"test mask directory: {test_mask_dir}")
    log.debug(f"batch_size: {batch_size}")
    log.debug(f"input image size: {input_image_size}")
    log.debug(f"output image size: {output_image_size}")

    def log_predictions(epoch, logs):
        '''
        callback function for logging predictions after every epoch.
        '''

        # define a folder for prediction logging
        file_writer_preds = tf_summary.create_file_writer(log_dir + '/predictions')
        
        # Generate another round of test batches.
        val_batches = test_batches_from_gtiff_dirs(test_img_dir, 
                                                     test_mask_dir, 
                                                     batch_size, 
                                                     input_image_size, 
                                                     output_image_size)

        # take a single batch from the data generator.
        # Note it has the same batch_size, height, and width, as the original image
        # Since this is a prediction problem with 2 classes, the prediction has 2 channels. 
        batch = next(val_batches.take(1).as_numpy_iterator())
        prediction = model.predict(batch[0])

        # make a mask from the prediction.
        predmask = get_mask_from_prediction(prediction)

        # display the result of training.
        img_to_log = plot_to_image(get_img_mask_pred_batch_figure(batch[0], batch[1], predmask))

        # Log the confusion matrix as an image summary.
        with file_writer_preds.as_default():
            tf_summary.image("Test Batch Predictions", img_to_log, step=epoch)
            
    return log_predictions