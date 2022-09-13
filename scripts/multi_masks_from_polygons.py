#################################################################
#
# created for ramp project, August 2022
# Author: carolyn.johnston@dev.global
#
#################################################################

 
 
import os, argparse, sys
from pathlib import Path


from ramp.utils.misc_ramp_utils import dir_path
from tqdm import tqdm
from ramp.data_mgmt.chip_label_pairs import get_tq_chip_label_pairs, construct_mask_filepath
from solaris.vector.mask import crs_is_metric
from solaris.utils.geo import get_crs
from solaris.utils.core import _check_rasterio_im_load
from ramp.utils.multimask_utils import df_to_px_mask, multimask_to_sparse_multimask, geojson_to_px_gdf, georegister_px_df, affine_transform_gdf
import rasterio as rio
from ramp.utils.img_utils import to_channels_last, to_channels_first
import numpy as np
import geopandas as gpd

# get rid of geopandas FutureWarning
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


# adding logging
# some options for log levels: INFO, WARNING, DEBUG
import logging
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
log.addHandler(ch)


def get_rasterio_shape_and_transform(image_path):
    '''
    get the image shape and the affine transform to pass into df_to_px_mask.
    '''
    with rio.open(image_path) as rio_dset:
        shape = rio_dset.shape
        transform = rio_dset.transform
    return shape, transform

def main():

    parser = argparse.ArgumentParser(description=''' 
    Create multichannel building footprint masks from a folder of geojson files. 
    This also requires the path to the matching image chips directory.

    Example: multi_masks_from_polygons.py -in_vecs labels_dir -in_chips chips_dir -out multimask_dir -bwidth 2 -csp 4

    This will produce 4-channel masks: backgrounds, building interiors, building borders, and close-contact points.
    The borders of the buildings will be 2 pixels wide (-bwidth 2).
    Pixels between buildings that are closer together than 4 pixels will be given a 'close-contact' class value (-csp 4).
    ''',formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-in_vecs', '--input_vec_dir', type=dir_path, required=True, help=r'Path to directory containing geojson files.')
    parser.add_argument('-in_chips', '--input_chip_dir', type = dir_path, required = True, help='Path to directory containing image chip files with names matching geojson files.')
    
    # made this directory a string, since it might not exist yet. If not, it will be created
    parser.add_argument('-out', '--output_dir', type=str, required=True, help=r'Path to directory containing output SDT masks.')
    parser.add_argument('-bwidth','--boundary_width', type=int, required=False, default=2, help=r"Width in pixel units of boundary class around building footprints, in pixels (optional: default=2)")
    parser.add_argument('-csp','--contact_spacing', type=int, required=False, default=4, help=r"pixels that are closer to two different polygons than contact_spacing (in pixel units) will be labeled with the contact mask (optional: default=4)")
    args = parser.parse_args()

    in_poly_dir = args.input_vec_dir
    in_chip_dir = args.input_chip_dir
    out_mask_dir = args.output_dir
    input_contact_spacing = args.contact_spacing
    input_boundary_width = args.boundary_width

    log.info(f'Input polygons directory: {in_poly_dir}')
    log.info(f'Input matching image chips directory: {in_chip_dir}')
    log.info(f'Output multichannel masks directory: {out_mask_dir}')
    log.info(f"Boundary mask boundary width: {input_boundary_width}")
    log.info(f"Contact mask contact spacing: {input_contact_spacing}")



    # If output mask directory doesn't exist, try to create it. 
    Path(out_mask_dir).mkdir(parents=True, exist_ok=True)
    
    chip_label_pairs = get_tq_chip_label_pairs(in_chip_dir, in_poly_dir)

    chip_paths, label_paths = list(zip(*chip_label_pairs))

    # construct the output mask file names from the chip file names.
    # these will have the same base filenames as the chip files, 
    # with a mask.tif extension in place of the .tif extension.
    mask_paths = [construct_mask_filepath(out_mask_dir, chip_path) for chip_path in chip_paths]
    
    # construct a list of full paths to the mask files
    json_chip_mask_zips = zip(label_paths, chip_paths, mask_paths)
    
    for json_path, chip_path, mask_path in tqdm(json_chip_mask_zips):

        # We will run this on very large directories, and some label files might fail to process.
        # We want to be able to resume mask creation from where we left off.         
        if Path(mask_path).is_file():
            continue

        log.debug(f"processing: {json_path}")
        
        # workaround for bug in solaris
        mask_shape, mask_transform = get_rasterio_shape_and_transform(chip_path)
        
        gdf = gpd.read_file(json_path)

        # remove empty and null geometries
        gdf = gdf[~gdf["geometry"].isna()]
        gdf = gdf[~gdf.is_empty]

        log.debug(f"Processing {json_path}: CRS is {get_crs(gdf)}, num_features: {len(gdf)}")

        reference_im = _check_rasterio_im_load(chip_path)
        
        if get_crs(gdf) != get_crs(reference_im):
        
            # BUGFIX: if crs's don't match, reproject the geodataframe
            gdf = gdf.to_crs(get_crs(reference_im))

        if crs_is_metric(gdf):
            meters=True

            # CJ 20220824: convert pixels to meters for call to df_to_pix_mask
            boundary_width = min(reference_im.res)*input_boundary_width
            contact_spacing = min(reference_im.res)*input_contact_spacing
        else:
            meters=False
            boundary_width=input_boundary_width
            contact_spacing=input_contact_spacing

        # NOTE: solaris does not support multipolygon geodataframes
        # So first we call explode() to turn multipolygons into polygon dataframes
        # ignore_index=True prevents polygons from the same multipolygon from being grouped into a series. -+
        gdf_poly = gdf.explode(ignore_index=True)

        # multi_mask is a one-hot, channels-last encoded mask
        onehot_multi_mask = df_to_px_mask(df=gdf_poly,
                                        out_file=mask_path,
                                        shape = mask_shape,
                                        do_transform = True,
                                        affine_obj = None,
                                        channels=['footprint', 'boundary', 'contact'],
                                        reference_im=reference_im,
                                        boundary_width=boundary_width,
                                        contact_spacing=contact_spacing, 
                                        out_type="uint8",
                                        meters=meters)


        # convert onehot_multi_mask to a sparse encoded mask 
        # of shape (1,H,W) for compatibility with rasterio writer
        sparse_multi_mask = multimask_to_sparse_multimask(onehot_multi_mask)
        sparse_multi_mask = to_channels_first(sparse_multi_mask)

        # write out sparse mask file with rasterio.
        with rio.open(chip_path, "r") as src:
            meta = src.meta.copy()
            meta.update(count=sparse_multi_mask.shape[0])
            meta.update(dtype='uint8')
            meta.update(nodata=None)
            with rio.open(mask_path, 'w', **meta) as dst:
                    dst.write(sparse_multi_mask)

    return 
            

if __name__ == "__main__":
    main()