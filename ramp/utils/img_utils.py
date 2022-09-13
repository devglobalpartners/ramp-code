#################################################################
#
# created for ramp project, August 2022
# Author: carolyn.johnston@dev.global
#
#################################################################


import numpy as np
from osgeo import gdal, gdalnumeric as gdn, gdalconst as gdc
from skimage.transform import resize
from .ramp_exceptions import GdalReadError

# adding logging
import logging
log = logging.getLogger(__name__)
log.addHandler(logging.NullHandler())

def to_channels_first(tensor_3):
    '''
    converts a channels-last to a channels-first shape.
    (i.e. from gdal to rasterio)
    '''
    return tensor_3.transpose(2,0,1)

def to_channels_last(tensor_3):
    '''
    converts a channels-first to a channels-last shape
    (from rasterio to gdal)
    '''
    return tensor_3.transpose(1,2,0)

### Detect whether an image has nodata values.

def nodata_count(the_image):
    '''
    image: Float32 np 3-tensor: assumed to be channels-last.
    returns true if the image has nodata values (ie, 0-0-0 at a pixel location)
    '''
    flat_image = the_image.transpose(2,0,1).reshape(3,-1)
    image_sz = flat_image.shape[1]
    inds = np.where((flat_image[0,:] == 0) & (flat_image[1,:] == 0) & (flat_image[2,:] == 0))
    return len(inds[0])


# Standardize and normalize

def standardize_rgb_tensor(the_tensor):
    '''
    rescale the input tensor to have (min, max) = (0,1).
    '''
    ftensor = the_tensor.astype(np.float32)
    the_max = np.max(ftensor)
    the_min = np.min(ftensor)
    span = the_max - the_min
    if span != 0.0:
        ftensor = (ftensor - the_min)/span
    else:
        ftensor = np.zeros(ftensor.shape)
    return ftensor


def normalize_std_tensor(std_tensor):
  '''
  Apply Imagenet normalization to a standardized (0-1) float tensor.  
  '''
  # check that the tensor has 3 channels
  assert std_tensor.shape[-1] == 3, f'unexpected tensor shape: {std_tensor.shape}'

  # apply normalization to each channel
  imgnet_mean = [0.485, 0.456, 0.406] 
  imgnet_stdev = [0.229, 0.224, 0.225]
  norm_tensor = np.zeros(std_tensor.shape)
  for ii in range(3):
    norm_tensor[...,ii] = (std_tensor[...,ii] - imgnet_mean[ii])/imgnet_stdev[ii]
  return norm_tensor


def convert_RGB_array_to_grayscale(rgb_array):
    '''
    converts a channels-last RGB array to a grayscale image using a widely accepted formula.
    :param np.ndarray rgb_array: channels-last RGB array as returned by gdal_get_image_tensor.  
    '''
    R = rgb_array[:,:,0]
    G = rgb_array[:,:,1]
    B = rgb_array[:,:,2]

    # standard conversion formula. Note the coefficients sum to 1.
    return 0.2989 * R  + 0.5870 * G + 0.1140 * B



def dhash(image, hashSize=8):
    '''
    dhash calculates a simple hash function of a channels-last RGB image.
    It's used for finding duplicates in large image datasets.
    The larger hashSize is, the larger the space of possible hashes 
    and the lower is the likelihood of 'hash-collisions' - 
    images that are different but have the same hash. 
    But keep in mind that the space of possible hash values for hashSize=8 is huge (2^64). 
    This is big enough for any reasonable purpose.

    :param np.ndarray image: channels-last RGB image
    '''
	
    # raise NotImplementedError("call to resize is failing in tensorflow, needs fixing")

    # convert the image to grayscale and resize the grayscale image,
	  # adding a single column (width).
    # Note that we are using tensorflow.image.resize(), which requires us to add 
    # a trivial 3rd dimension (using np.newaxis) to the grayscale image before passing it.
    gray = convert_RGB_array_to_grayscale(image)
    # gray_3band = gray[:,:,np.newaxis]
    resized = resize(gray, (hashSize + 1, hashSize), order=0)


	# compute the (relative) horizontal gradient between adjacent
	# column pixels. The result is an array of 0s and 1s of size (hashsize, hashsize)
    diff = resized[:, 1:] > resized[:, :-1]

	# convert the difference image to a hash (integer) and return it
    return sum([2 ** i for (i, v) in enumerate(diff.flatten()) if v])


############################################################
# reading and writing raster geotiff files 
############################################################

def gdal_get_dataset(filename):
  '''
  Open gdal-readable file, return dataset
  '''
  ds  = gdal.Open(filename)
  if not ds:
      log.error(f"GDAL could not read {filename}")
      raise GdalReadError(f"Could not read {filename}")
  return ds


def gdal_get_mask_tensor(mask_path):
    '''
    Reads a mask geotiff image using gdal, and returns a numpy tensor containing the mask.
    Masks have datatype uint8, and a single channel.
    mask_path: path to a gdal-readable image.
    
    Channels go last in the index order for the output tensor, as is standard for tensorflow.

    :param str mask_path: the path to the mask file
    :param str its_dtype: datatype of numpy mask tensor to create. If None, will be inferred from the GDAL datatype.

    returns: a numpy ndarray.
    '''

    ds  = gdal.Open(mask_path)
    if not ds:
      log.error(f"GDAL could not read {mask_path}")
      raise GdalReadError(f"Could not read {mask_path}")

    mask_tensor = get_mask_tensor_from_gdal_dataset(ds)
    return mask_tensor



def gdal_get_image_tensor(image_path):    
  '''
  Reads a satellite image geotiff using gdal, and converts to a numpy tensor.
  Image geotiffs have 3 channels (RGB) and are of type float32.
  image_path: path to a gdal-readable color image.
  
  Channels go last in the index order for the output numpy array.

  :param str image_path: the path to the image file. 
  '''

  its_dtype = "float32"
  ds  = gdal.Open(image_path)
  if not ds:
    log.error(f"GDAL could not read {image_path}")
    raise GdalReadError(f"Could not read {image_path}")
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

# 2022.04.29 added requirement to specify gdal data type
def gdal_write_mask_tensor(mask_path, src_ds, mask_tensor, gdal_dtype):
  '''
  
  Writes a mask tensor to a new file, with georeferencing matching the gdal dataset.
  The data type of the file to be written depends on the datatype of the mask tensor, which may be either byte or int32.

  :param filename mask_path: path to mask to be created
  :param GdalDataset src_ds: gdal dataset with georeferencing for the mask tensor to match.
  :mask_tensor numpy.ndarray: numpy array containing mask values. Its size should match src_ds or an AssertionError is thrown.
  :gdal_dtype gdal datatype: data type of gdal dataset, usually one of gdal.GDT_Byte, gdal.GDT_Int32, gdal.GDT_Float32
  '''

 # this should create the dataset, then flush it and write out the mask file. 
  mask_ds = create_gdal_dataset_from_mask_tensor(src_ds, mask_tensor, 
                                                  in_ram=False, 
                                                  mask_path=mask_path)
  mask_ds = None
  return


############################################################
# working with GDAL datasets
############################################################

def get_mask_tensor_from_gdal_dataset(ds):
  '''
  extract a tensor from a 1-channel gdal raster dataset.
  :param ds GdalDataset: raster dataset containing the mask

  returns: a numpy ndarray of uint8 datatype
  '''
  assert ds is not None, 'null dataset'

  bands = [ds.GetRasterBand(i) for i in range(1, ds.RasterCount + 1)]
    
  # create a channels_last byte array -- rows, columns
  its_dtype=np.byte
  cols = ds.RasterXSize 
  rows = ds.RasterYSize 
  the_array = np.zeros((rows, cols, ds.RasterCount), its_dtype)
  
  for bnum, the_band in enumerate(bands):
    from_img = gdn.BandReadAsArray(the_band)
    to_img = the_array[:,:,bnum]
    np.copyto(to_img, from_img)
  return the_array


# 2022.04.29 added requirement to specify gdal data type
# 2022.05.20: removed gdal data type requirement (and gdal_dtype input parameter): all masks are byte
def create_gdal_dataset_from_mask_tensor(src_ds, mask_tensor, in_ram=True, mask_path=None):
  '''
  Creates an in-ram or geotiff mask GdalDataset using the mask tensor, with georeferencing matching the gdal dataset.
  In-ram is the default. If in_ram=False is passed, the mask_path must be given.

  :param GdalDataset src_ds: gdal dataset with georeferencing for the mask tensor to match.
  :param filename mask_path: path to mask to be created
  :param mask_tensor numpy.ndarray: numpy array containing mask values. Its size should match src_ds or an AssertionError is thrown.
  :param gdal datatype gdal_dtype: usually one of gdal.GDT_Byte, gdal.GDT_Int32, gdal.GDTFloat32
  Returns the gdal dataset. 
  To flush a geotiff mask to file, set it to None in the calling code. (rolleyes) 
  '''

  gdal_dtype = gdal.GDT_Byte

  # check mask shape 
  mask_shape = mask_tensor.shape
  mask_dims = len(mask_shape)
  sz_x, sz_y = src_ds.RasterXSize, src_ds.RasterYSize
  assert mask_dims == 2 or (mask_dims == 3 and mask_shape[2] == 1), 'mask is neither a 2-tensor nor a 3-tensor with a single channel'

  # if mask is a 3-tensor with a single channel, change it to a 2-tensor (22.04.29: using np.squeeze)
  # No op if it's a 2-tensor
  mask_tensor = np.squeeze(mask_tensor) 
 
  # check reference file dimensions vs. mask tensor dimensions
  # these are reversed in the create functions for gdal vs numpy
  assert mask_tensor.shape[0] == sz_y and mask_tensor.shape[1] == sz_x, \
  'reference image and mask to be created have different shapes'

  # Create a mask raster in memory, matching the parameters of the geotiff image (src_ds).
  # Note: when creating gdal images, the order of dimensions is sz_x, then sz_y
  sz_x, sz_y = src_ds.RasterXSize, src_ds.RasterYSize
  driver = None
  mask_ds = None
  if in_ram:
    driver = gdal.GetDriverByName('MEM')
    mask_ds = driver.Create('',sz_x, sz_y, 1, gdal_dtype)
  else:
    assert mask_path is not None, 'mask path was not given'
    driver = gdal.GetDriverByName('GTiff')
    mask_ds = driver.Create(mask_path,sz_x, sz_y, 1, gdal_dtype)
  assert mask_ds is not None, f'Mask file {mask_path} could not be created'

  mask_ds.SetGeoTransform(src_ds.GetGeoTransform())
  mask_ds.SetProjection(src_ds.GetProjection())
  
  # insert mask array into gdal ds
  mask_ds.GetRasterBand(1).WriteArray(mask_tensor)
  
  # I think those are all the magic incantations you need. 
  return mask_ds


############################################################
# get binary mask, given vector layer and reference GDAL Dataset
############################################################

def get_binary_mask_from_layer(layer, src_ds, in_ram=True, mask_path=None):
  '''
  given a polygon layer and a reference GDAL dataset, 
  create a new binary byte mask and burn the raster into it.
  By default, the mask will be a gdal dataset in ram.
  '''
  # Create a byte mask GDAL dataset, matching the parameters of the geotiff image (src_ds).

  sz_x, sz_y = src_ds.RasterXSize, src_ds.RasterYSize
  blank_array = np.zeros((sz_y, sz_x), dtype = np.byte)
  mask_ds = create_gdal_dataset_from_mask_tensor(src_ds, blank_array, in_ram, mask_path)
  
  # Rasterize the feature into the in-ram mask  
  band = mask_ds.GetRasterBand(1)
  gdal.RasterizeLayer(mask_ds, [1], layer, burn_values=[1])
  band.FlushCache()
  
  # call ReadAsArray on the binary mask raster to get it as a np array.
  mask_array = mask_ds.ReadAsArray()

  # free the mask dataset, or flush the mask to a file if it's not in-ram
  mask_ds = None
  return mask_array
  

############################################################
# get nodata mask from an image
############################################################

def get_nodata_mask_from_image(image_tensor):
  '''
  image_tensor: RGB (float) numpy array of shape (width, height, 3)
  returns mask_tensor: byte numpy array of shape (width, height, 1)
  '''
  mask_tensor_0 = image_tensor[:,:,0]==0.0  
  mask_tensor_1 = image_tensor[:,:,1]==0.0  
  mask_tensor_2 = image_tensor[:,:,2]==0.0

  log.info(f"nodata mask shape: {mask_tensor_2.shape}")
  return_mask = np.logical_and(mask_tensor_0, mask_tensor_1, mask_tensor_2).astype(np.byte)
  return return_mask
  