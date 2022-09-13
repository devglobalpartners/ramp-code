#################################################################
#
# created for ramp project, August 2022
# Author: carolyn.johnston@dev.global
#
#################################################################



from .img_utils import create_gdal_dataset_from_mask_tensor
from osgeo import gdal, ogr, osr
import numpy as np

def onehot_to_sparse_multimask(onehot_mask):
    """
    Change a one-hot (channels-last) mask to a sparse-encoded mask
    onehot_mask: numpy ndarray of shape [H,W, C] with values 0,1
    sparse_mask: numpy ndarray of shape [H,W,1] with values 0,...,C-1 
    """
    sparse_mask = np.zeros(onehot_mask.shape[:2], dtype=np.uint8)
    sparse_mask[np.where(onehot_mask[:,:,0] > 0)] = 1
    sparse_mask[np.where(onehot_mask[:,:,1] > 0)] = 2
    sparse_mask[np.where(onehot_mask[:,:,2] > 0)] = 3
    return np.expand_dims(sparse_mask, axis=-1)

def sparse_multimask_to_onehot_mask(num_classes, sparse_mask):
    """
    NOT TESTED and probably buggy
    Change a sparse-encoded to a onehot (channels-last) mask
    num_classes: C
    sparse_mask: numpy ndarray of shape [H,W,1] with values in 0,...,C-1 
    onehot_mask: numpy ndarray of shape [H,W, C] with values 0,1
    """

    raise NotImplementedError

    # get H and W
    dims = sparse_mask.shape[:2]

    # get rid of 3rd trivial mask dimension if present
    if len(sparse_mask.shape) == 3:
        sparse_mask = np.squeeze(sparse_mask, axis=-1)

    indices = np.indices(sparse_mask.shape)

    # construct onehot mask
    onehot_mask = np.zeros((*dims, num_classes), dtype = np.uint8)
    onehot_mask[indices[0],indices[1], sparse_mask]=1
    return onehot_mask

def binary_mask_from_multichannel_mask(multimask):
    '''
    create a binary building mask from a multichannel prediction.
    :param ndarray multimask: 4-channel, sparse-encoded multichannel mask
    (classes: background:0, building:1, boundary:2, close-contact-point:3)
    '''

    # aggregate building points, remaining points are 0
    # 20220808: can't include boundary points, in tight places you get blobs.
    foregd_mask = (multimask == 1)
    return foregd_mask.astype(np.uint8)


def binary_mask_from_sdt_mask(sdt_mask, bldg_lower_thresh):
    '''
    create a binary building mask from a prediction based on the Truncated Signed Distance Transform.
    Right now this process is a simple thresholding. 
    We'll add more post-processing functionality here as necessary. 

    :param ndarray sdt_mask: int32 truncated signed distance transform mask.
    :param int bldg_lower_thresh: threshold below which the mask pixel will be set to 0. 
    '''
    mask = sdt_mask >= bldg_lower_thresh
    return mask.astype('uint8')

def binary_mask_to_geojson(mask, reference_ds, geojson_path):
    '''

    Polygonize a binary mask to geojson file.
    I wanted to just print it out to a string, but can't figure out how to do this:
    OGR triumphs again.

    :param ndarray mask: numpy ndarray containing the binary byte mask to polygonize.
    :param reference_ds: gdal dataset containing the georeferencing to use for the output geojson. 
    :param geojson_path: path to geojson file for exported data.
    '''

    # insert binary mask into an in-memory gdal dataset with the same georeferencing as the reference_ds.
    # this should be an img_utils function call. 
    mask_ds = create_gdal_dataset_from_mask_tensor(reference_ds, mask)
    
    # create a geojson layer to polygonize into.
    # this as usual is a pita in OGR. 
    drv = ogr.GetDriverByName('GeoJSON')
    vec_ds = drv.CreateDataSource(geojson_path)
    dst_layername = 'mask'
    prj = reference_ds.GetProjection()
    
    srs = osr.SpatialReference(wkt=prj)
    layer = vec_ds.CreateLayer(dst_layername, srs = srs, geom_type = ogr.wkbPolygon )
    
    # gdal.Polygonize will want to write the pixel value of the connected component
    # into an attribute in the layer. You need to create this attribute.
    # if you don't, the code will run anyway, but OGR will complain. 
    # this is very poorly documented online. 
    # explanation here: https://gdal.org/api/gdal_alg.html#_CPPv414GDALPolygonize15GDALRasterBandH15GDALRasterBandH9OGRLayerHiPPc16GDALProgressFuncPv
    pixValueField = ogr.FieldDefn( 'pixel_value', ogr.OFTInteger )
    layer.CreateField( pixValueField )

    # get the mask, and call polygonize. 
    # no one online seems clear on what that 4th parameter does. 
    band = mask_ds.GetRasterBand(1)

    # that second parameter causes gdal to use the same band for a mask, to determine which connected 
    # regions to polygonize. 0 areas (background!) are not polygonized. Otherwise, if mask is None, 
    # the background gets polygonized, not just the buildings.
    # gdal.Polygonize(band, None, layer, 0, [])
    gdal.Polygonize(band, band, layer, 0, [])
    # gdal.Polygonize(band, None, layer, 0, [])

    # and brilliant: here's how you save and close the geojson file you created. 
    vec_ds = None
    return
    



def sdt_mask_to_geojson(sdt_mask, reference_ds, geojson_path, bldg_lower_thresh=3):
    '''
    convert truncated sdt mask to a geojson and writes it to a file.

    :param ndarray sdt_mask: byte mask derived from truncated signed distance transform.
    :param GdalDataset reference_ds: dataset from which to extract georeferencing (usually the geotiff image chip).
    :param int bldg_lower_threshold: threshold below which a mask pixel is considered background and set to 0.
    '''

    # create binary mask. 
    bin_mask = binary_mask_from_sdt_mask(sdt_mask, bldg_lower_thresh)
    binary_mask_to_geojson(bin_mask, reference_ds, geojson_path)
    return
