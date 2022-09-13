
#################################################################
#
# created for ramp project, August 2022
# Author: carolyn.johnston@dev.global
#
#################################################################

import tensorflow as tf
from tensorflow.keras.layers.experimental import preprocessing 
import os
from osgeo import gdal, gdalnumeric as gdn
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.python.ops.gen_batch_ops import batch
from ..utils.ramp_exceptions import GdalReadError
import random

# adding logging
import logging
log = logging.getLogger(__name__)
log.addHandler(logging.NullHandler())

###########################################################################################

def tf_gdal_get_image_tensor(image_path):    
  '''
  Reads a satellite image geotiff using gdal, and converts to a numpy tensor.
  Image geotiffs have 3 channels (RGB) and are of type float32.

  
  Channels go last in the index order for the output tensor, as is standard for tensorflow.

  :param tf.StringTensor image_path: path to the image file, as a tensorflow String Tensor.
  '''

  its_dtype = "float32"
  ds  = gdal.Open(image_path.numpy())
  if not ds:
      log.error(f"GDAL could not read {image_path.numpy()}")
      raise GdalReadError(f"Could not read {image_path.numpy()}")
  bands = [ds.GetRasterBand(i) for i in range(1, ds.RasterCount + 1)]
  cols = ds.RasterXSize 
  rows = ds.RasterYSize 
    
  # create a channels_last array -- rows, columns
  the_array = np.zeros((rows, cols, ds.RasterCount), its_dtype)
  
  for bnum, the_band in enumerate(bands):
    from_img = gdn.BandReadAsArray(the_band)
    to_img = the_array[:,:,bnum]
    np.copyto(to_img, from_img)
    
  # normalize the float image to [0,1]
  return the_array/np.max(the_array)


def tf_gdal_get_image_tensor_with_name(image_path):

  return tf_gdal_get_image_tensor(image_path), image_path

###########################################################################################

def tf_gdal_get_mask_tensor(mask_path):
  '''
  Reads a mask geotiff image using gdal, and converts to a numpy tensor.
  Masks have datatype uint8, and a single channel.
  mask_path: path to a gdal-readable image.
  
  Channels go last in the index order for the output tensor, as is standard for tensorflow.

  :param tf.StringTensor mask_path: path to the image file, as a tensorflow String Tensor.
  '''

  its_dtype = "uint8"

  ds  = gdal.Open(mask_path.numpy())
  if not ds:
    log.error(f"GDAL could not read {mask_path.numpy()}")
    raise GdalReadError(f"Could not read {mask_path.numpy()}")
  bands = [ds.GetRasterBand(i) for i in range(1, ds.RasterCount + 1)]
  cols = ds.RasterXSize 
  rows = ds.RasterYSize 
    
  # create a channels_last array -- rows, columns
  the_array = np.zeros((rows, cols, ds.RasterCount), its_dtype)
  
  for bnum, the_band in enumerate(bands):
    from_img = gdn.BandReadAsArray(the_band)
    to_img = the_array[:,:,bnum]
    np.copyto(to_img, from_img)
  return the_array

def tf_gdal_get_mask_tensor_with_name(mask_path):

  return tf_gdal_get_mask_tensor(mask_path), mask_path

###########################################################################################

def resize_it(an_image, original_shape, new_size, the_method):
    '''
    Used with 'map' to resize images in the data pipeline.
    an_image: an image array of shape 'original shape'.
    original_shape: the shape of the input image array.
    new_size: the new dimensions (width, height) of the output image array. 
        The number of channels is unchanged, so is not specified in 'new_size'.
    the_method: method specified for interpolation. 'nearest' for masks, 'bilinear' for images. 
    '''

    # this call to set_shape works around a bug in tensorflow
    # the bug is described in this article: 
    # https://stackoverflow.com/questions/62957726/i-got-value-error-that-image-has-no-shape-while-converting-image-to-tensor-for-p
    an_image.set_shape(original_shape)
    return tf.image.resize(an_image, new_size, method=the_method)

def resize_it_with_name(an_image, original_shape, new_size, the_method, pathname):

  return resize_it(an_image, original_shape, new_size, the_method), pathname


  # AUGMENTATION ADDITIONS
  ###########################################################################################

# AUG
import albumentations as A
from functools import partial



### APPLY AUGMENTATION WORKAROUND

# This is a function that prepackages another function that actually applies augmentation.
# This workaround is needed because I can't pass img_shape and transforms to tf.numpy_function.

def get_apply_augmentation_function(img_shape, transforms):
  '''
  Function that prepackages a function that applies augmentation.
  Needed because I can't pass img_shape and transforms to tf.numpy_function.
  '''

  # define prepackaged function to apply augmentation, using the given albumentation transform.
  def apply_augmentation(image, mask):
    '''
    Apply augmentation using the albumentations library.
    image: ndarray (image tensor).
    mask: ndarray (mask tensor)

    Packaged parameters:
    img_shape: shape to use in call to tf.image.resize, following the transform.
    #todo: can you replace this with a call to resize_it?
    transforms: the augmentation object returned from call to albumentations.Compose().
    returns: augmented image and mask.   

    Code reference page: https://albumentations.ai/docs/examples/tensorflow-example/
    Note: certain augmentations are applied only to the mask.
    See the bottom of this webpage: https://github.com/albumentations-team/albumentations_examples/blob/master/notebooks/example_kaggle_salt.ipynb.  
    '''
    data = {"image":image, "mask":mask}
    
    if transforms is not None:
      aug_data = transforms(**data)
      aug_img = aug_data["image"]
      aug_mask = aug_data["mask"]
    else:
      aug_img = image
      aug_mask = mask

    ## DEBUG
    # check if mask was transformed
    # changed = aug_mask != mask
    # print(f'Mask was transformed: {changed.any()}')

    # not needed -- aug_img is already normalized to the range 0-1
    # renormalization might be needed once contrast/brightness shifts are added to augmentations
    # aug_img = tf.cast(aug_img/255.0, tf.float32)
    aug_mask = tf.cast(aug_mask, tf.uint8)
    aug_img = tf.image.resize(aug_img, size=img_shape)
    aug_mask = tf.image.resize(aug_mask, size=img_shape, method='nearest')
    return aug_img, aug_mask

  # return the prepackaged function
  return apply_augmentation
### END APPLY AUGMENTATION WORKAROUND

def process_data(image, mask, img_shape, transforms):
  '''
  Code reference page: https://albumentations.ai/docs/examples/tensorflow-example/
  '''
  apply_aug_fn = get_apply_augmentation_function(img_shape, transforms)
  aug_img, aug_mask = tf.numpy_function(func = apply_aug_fn, inp = [image, mask], Tout=[tf.float32, tf.uint8])
  #print(aug_img.dtype, aug_mask.dtype)
  return aug_img, aug_mask

def reset_shapes(image, mask, image_shape):
  image.set_shape([*image_shape, 3])
  mask.set_shape([*image_shape,1])
  return image, mask

  # END AUGMENTATION ADDITIONS
  ###########################################################################################


def training_batches_from_gtiff_dirs(
    image_file_dir,
    mask_file_dir,
    batch_size, 
    input_image_size,
    output_image_size,
    transforms = None):
    '''
    image_file_dir: a directory containing geotiff image files to be used in training.
    mask_file_dir: a directory containing matching geotiff mask files to be used in training.
    batch_size: batch size, a positive integer.
    input_image_size: dimensions of the input images in the form (rows, columns).
    output_image_size: dimensions of the output images in the form (new_rows, new_columns).
    tfgseed: Tensorflow global random seed for reproducible results. 
    transforms: Albumentations object for augmenting training data. None == no augmentation (default).

    Matching images and masks are assumed to have names of the form:
    unique_string.tif, same_unique_string.mask.tif. 
    
    Examples of matching image and mask: 
    "training_img909.tif",
    "training_img909.mask.tif".

    Example input image size: (650, 650).
    Example output image size: (512, 512).

    Returns: a tf.data.Dataset containing batches of matching images and masks, \
      suitable for using in model.fit. 
   '''

    tf_global_seed = 20220607
    py_seed = 303041

    # set this for reproducibility in file listings
    tf.random.set_seed(tf_global_seed)

    # set this for reproducibility in augmentations
    random.seed(py_seed)

    image_pat = os.path.join(image_file_dir, '*.tif')
    mask_pat = os.path.join(mask_file_dir, '*.mask.tif')


    # giving these the same op-level random seed causes them to be listed in the same order.
    image_ds = tf.data.Dataset.list_files(image_pat, seed=333)
    mask_ds = tf.data.Dataset.list_files(mask_pat, seed=333)

    # get the number of training images and masks. Since they match, the numbers should be the same. 
    n_images = tf.data.experimental.cardinality(image_ds)
    n_masks = tf.data.experimental.cardinality(mask_ds)
    assert n_images==n_masks, f"Number of images: {n_images} | Number of masks: {n_masks}"
    
    # Define functions to use for loading gdal images in the data pipeline.
    # These are just Tensorflow wrappers around 
    # the gdal_get_image_tensor and gdal_get_mask_tensor functions.
    tf_load_image_fn = lambda imgname: tf.py_function(func = tf_gdal_get_image_tensor, 
                              inp=[imgname], 
                              Tout=[tf.float32])

    tf_load_mask_fn = lambda maskname: tf.py_function(func = tf_gdal_get_mask_tensor,
                                                   inp = [maskname],
                                                    Tout = [tf.uint8])
    
    # resize the images and masks to the desired size.
    images = image_ds.map(tf_load_image_fn).map(lambda x: resize_it(x, [*input_image_size, 3], output_image_size, 'bilinear'))
    masks = mask_ds.map(tf_load_mask_fn).map(lambda x: resize_it(x, [*input_image_size, 1], output_image_size,'nearest'))

    # zip them together so the resulting dataset puts out a data element of the form (image, matching mask).
    zip_pairs = tf.data.Dataset.zip((images, masks), name=None)

    len_train = n_images
    buffer_size = len_train

    data_batches = None
    
    if transforms is not None:
      data_batches = (
      zip_pairs
      .cache()
      .shuffle(buffer_size)
      # START augmentation 
      # the next two lines apply the augmentation. 
      # The dataset loses its shape after applying a tf.numpy_function, 
      # so it's necessary to call reset_shapes for the sequential model and when inheriting from the model class.
      # See 'restoring dataset shapes' in https://albumentations.ai/docs/examples/tensorflow-example/.
      .map(partial(process_data, img_shape=output_image_size, transforms=transforms),
                      num_parallel_calls=tf.data.AUTOTUNE)
      .map(partial(reset_shapes, image_shape=output_image_size), 
                      num_parallel_calls=tf.data.AUTOTUNE)
      # END augmentation
      .batch(batch_size)
      .repeat()
      .prefetch(buffer_size=tf.data.AUTOTUNE)
      )


    else:
      data_batches = (
        zip_pairs
        .cache()
        .shuffle(buffer_size)
        .batch(batch_size)
        .repeat()
        .prefetch(buffer_size=tf.data.AUTOTUNE)
        )

    return data_batches
    
###########################################################################################


def test_batches_from_gtiff_dirs(
    image_file_dir,
    mask_file_dir,
    batch_size,
    input_image_size,
    output_image_size
    ):
      '''

      Data generator for test batches.
      This data generator does not shuffle, cache, repeat, or augment -- it just delivers all
      test image batches in the same order every time.

      See the notes for 'train_batch_from_gtiff_dirs' for more details.
      '''

      image_pat = os.path.join(image_file_dir, '*.tif')
      mask_pat = os.path.join(mask_file_dir, '*.tif')


      # giving these the same random seed causes them to be listed in the same order.
      image_ds = tf.data.Dataset.list_files(image_pat, seed=333)
      mask_ds = tf.data.Dataset.list_files(mask_pat, seed=333)

      # get the number of training images and masks. Since they match, the numbers should be the same.
      # todo: make the data generator end appropriately if the numbers don't match.
      n_images = tf.data.experimental.cardinality(image_ds)
      n_masks = tf.data.experimental.cardinality(mask_ds)
      assert n_images==n_masks, f"Num images: {n_images}, Num masks: {n_masks} "

      # Define functions to use for loading gdal images in the data pipeline.
      # These are just Tensorflow wrappers around
      # the gdal_get_image_tensor and gdal_get_mask_tensor functions.
      tf_load_image_fn = lambda imgname: tf.py_function(func = tf_gdal_get_image_tensor,
                                inp=[imgname],
                                Tout=[tf.float32])

      tf_load_mask_fn = lambda maskname: tf.py_function(func = tf_gdal_get_mask_tensor,
                                                    inp = [maskname],
                                                      Tout = [tf.uint8])

      # resize the images and masks to the desired size.
      images = image_ds.map(tf_load_image_fn).map(lambda x: resize_it(x, [*input_image_size, 3], output_image_size, 'bilinear'))
      masks = mask_ds.map(tf_load_mask_fn).map(lambda x: resize_it(x, (*input_image_size, 1), output_image_size,'nearest'))

      # zip them together so the resulting dataset puts out a data element of the form (image, matching mask).
      zip_pairs = tf.data.Dataset.zip((images, masks), name=None)

      len_train = n_images
      buffer_size = len_train

      data_batches = (
        zip_pairs.batch(batch_size)
      )

      return data_batches

###################################################################

def test_batches_from_gtiff_dirs_with_names(
    image_file_dir,
    mask_file_dir,
    batch_size,
    input_image_size,
    output_image_size
    ):
      '''

      Data generator for test batches.
      This data generator does not shuffle, cache, repeat, or augment -- it just delivers all
      test image batches in the same order every time.

      See the notes for 'train_batch_from_gtiff_dirs' for more details.
      '''

      image_pat = os.path.join(image_file_dir, '*.tif')
      mask_pat = os.path.join(mask_file_dir, '*.tif')


      # giving these the same random seed causes them to be listed in the same order.
      image_ds = tf.data.Dataset.list_files(image_pat, seed=333)
      mask_ds = tf.data.Dataset.list_files(mask_pat, seed=333)

      # get the number of training images and masks. Since they match, the numbers should be the same.
      # todo: make the data generator end appropriately if the numbers don't match.
      n_images = tf.data.experimental.cardinality(image_ds)
      n_masks = tf.data.experimental.cardinality(mask_ds)
      assert n_images==n_masks, f"Num images: {n_images}, Num masks: {n_masks} "

      # Define functions to use for loading gdal images in the data pipeline.
      # These are just Tensorflow wrappers around
      # the gdal_get_image_tensor and gdal_get_mask_tensor functions.
      tf_load_image_fn = lambda imgname: tf.py_function(func = tf_gdal_get_image_tensor_with_name,
                                inp=[imgname],
                                Tout=[tf.float32, tf.string])

      tf_load_mask_fn = lambda maskname: tf.py_function(func = tf_gdal_get_mask_tensor_with_name,
                                                    inp = [maskname],
                                                      Tout = [tf.uint8, tf.string])

      # resize the images and masks to the desired size.
      images = image_ds.map(tf_load_image_fn).map(lambda x, pathname: resize_it_with_name(x, [*input_image_size, 3], output_image_size, 'bilinear', pathname))
      masks = mask_ds.map(tf_load_mask_fn).map(lambda x, pathname: resize_it_with_name(x, (*input_image_size, 1), output_image_size,'nearest', pathname))

      # zip them together so the resulting dataset puts out a data element of the form (image, matching mask).
      zip_pairs = tf.data.Dataset.zip((images, masks), name=None)

      len_train = n_images
      buffer_size = len_train

      data_batches = (
        zip_pairs.batch(batch_size)
      )

      return data_batches
      

###########################################################################################
## AUGUST 3rd 2022: modifications for using weighted loss functions
## for details on this approach to using class weights in loss computation see this site:
## https://www.tensorflow.org/tutorials/images/segmentation#optional_imbalanced_classes_and_class_weights 
###########################################################################################

def add_class_weights(chip, label, class_weights):
  ''' 
  create a class weight image with the same shape as the 'label' tensor,
  and return it along with the chip and label. 
  class_weights: a float tensorflow (not numpy) array of length equal to the number of classes,
  with weight values to be applied in the loss computation for each class.
  '''

  # normalize the class weights so they sum to 1
  class_weights = class_weights/tf.reduce_sum(class_weights)

  # create the weight tensor. 'gather' works by creating a tensor the same size as the 
  # 'label' object, and using the class index at each pixel of the label to index into the class_weights values.
  weight_tensor = tf.gather(class_weights, tf.cast(label, tf.int32))
  return chip, label, weight_tensor