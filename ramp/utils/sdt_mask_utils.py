#################################################################
#
# created for ramp project, August 2022
# Author: carolyn.johnston@dev.global
#
#################################################################




import numpy as np
import skfmm
from osgeo import gdal, ogr

# adding logging
import logging

from .img_utils import get_binary_mask_from_layer
log = logging.getLogger(__name__)
log.addHandler(logging.NullHandler())

def GetTruncatedSDT(layer, src_ds, m_per_pixel, clip_vals):
    '''
    Calculate the (unquantized) SDT from a binary mask created from an OGR layer.

    :param ogr.Layer layer: ogr (buildings) layer from geojson or shapefile
    :param GdalDataset src_ds: raster matching input vector layer (SDT will match its shape)
    :param m_per_pixel float: meters per pixel of src_ds
    :param tuple clip_vals: clip SDT values outside this range to the range's edges

    returns: np.ndarray containing the truncated (float) SDT, pre-quantization.
    '''

    # 2022.02.16 refactored to use img_utils.get_binary_mask_from_layer()
    bin_mask = get_binary_mask_from_layer(layer, src_ds)
    
    # Create mask using signed distance function    
    fmm_arr = np.where(bin_mask >= 1, 1, -1)  # Set all values to 1 or -1

    # constant arrays have no zero contour
    # if fmm_arr is constant, return an array with 
    # the minimum possible value of distance 
    # so it has no impact on the final truncated SDT
    if np.all(fmm_arr == fmm_arr[0,0]):
        return np.full(bin_mask.shape, clip_vals[0], dtype=np.int32)
        
    # call skfmm.distance() on it to get a signed distance transform.
    # truncate the SDT to the clip range.
    dist_arr = skfmm.distance(fmm_arr, dx=m_per_pixel).clip(clip_vals[0],clip_vals[1])
        
    # return the (unquantized) signed distance transform array
    return dist_arr

    
    


def QuantizeSDT(the_sdt):
    '''
    quantizes the SDT and returns an Int32 segmentation mask.
    '''
    return np.rint(the_sdt).astype(np.int32)


def CreateModifiedSDTMask(layer, src_ds, m_per_pixel, clip_vals):
    '''
    Creates a Truncated SDT that treats buildings with a common edge as separate buildings.
    :param ogr.Layer layer: ogr (buildings) layer from geojson or shapefile
    :param GdalDataset src_ds: raster matching input vector layer (the returned SDT will match its shape)
    '''
    
    # create tensor to hold results
    num_features = layer.GetFeatureCount()

    log.info(f'Number of features in layer: {num_features}')
    sz_x, sz_y = src_ds.RasterXSize, src_ds.RasterYSize

    if num_features == 0:
        # return a blank mask
        return np.zeros((sz_x, sz_y)).astype(np.int32)

    sdt_array = np.zeros((sz_x, sz_y, num_features)).astype(np.int32)

    # layer is an iterator with mysterious OGR behaviors. 
    # I reset it alot to be sure it is pointed at the head of the feature list.
    layer.ResetReading()

    # make an editable copy of the layer
    # this as usual is a pita in OGR. 
    # Look at this webpage for details: 
    # http://geoexamples.blogspot.com/2012/01/creating-files-in-ogr-and-gdal-with.html
    drv = ogr.GetDriverByName('Memory')
    ram_ds = drv.CreateDataSource( "out" )
    layer_copy = ram_ds.CopyLayer(layer, "copy")
    layer.ResetReading()

    # remove all features in the layer copy
    for feature in layer_copy:
        layer_copy.DeleteFeature(feature.GetFID())

    # check that layer_copy is now empty
    # print(f"Layer_copy feature count: {layer_copy.GetFeatureCount()}")

    # iterate over features in original layer, creating the SDT
    findex = 0

    for feature in layer:
        # create a truncated SDT with the single feature and store it in the ndarray
        layer_copy.CreateFeature(feature)
        sdt_array[:,:,findex] = QuantizeSDT(GetTruncatedSDT(layer_copy, src_ds, m_per_pixel, clip_vals))
        # print(f"Debug: {np.max(sdt_array[:,:,findex])}")
        findex = findex + 1
        layer_copy.DeleteFeature(feature.GetFID())
    layer.ResetReading()

    # take the pixelwise max over the stack of SDTs and return it
    final_SDT = sdt_array.max(axis=2)

    # clean up 
    ram_ds = None

    return final_SDT

def ProcessLabelToSdtMask(json_path, src_ds, mpp, clip_vals):
    
    log.info(f"Processing label file: {json_path}")
    drv = ogr.GetDriverByName('GeoJSON')
    ogr_ds = drv.Open(json_path)
    lyr = ogr_ds.GetLayer()

    # create the truncated signed distance transform mask
    sdt_array = CreateModifiedSDTMask(lyr, src_ds, mpp, clip_vals)

    # added 20220520: adjust mask to have values in the range [0,numclasses)
    sdt_array = sdt_array - clip_vals[0]
    return sdt_array
