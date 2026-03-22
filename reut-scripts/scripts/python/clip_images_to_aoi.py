#################################################################
#
# created for ramp project, August 2022
# Author: carolyn.johnston@dev.global
#
#################################################################

 
 
import os, argparse, sys
from pathlib import Path

# get rid of geopandas FutureWarning
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


from ramp.utils.misc_ramp_utils import dir_path
from ramp.utils.ramp_exceptions import GdalReadError
from ramp.utils.multimask_utils import binary_mask_to_geodataframe, buffer_df_geoms
from ramp.utils.mask_to_vec_utils import binary_mask_from_multichannel_mask
from ramp.utils.geo_utils import get_polygon_indices_to_merge, fill_rings, gpd_add_area_perim

import fiona

import rasterio as rio
from rasterio.warp import calculate_default_transform, reproject, Resampling

import rasterio.mask

import geopandas as gpd
import pandas as pd
import numpy as np
from tqdm import tqdm

# adding logging at info level
# script will reset logging at debug level if --debug switch is used
import logging
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

def main():

    parser = argparse.ArgumentParser(description=''' 
    Clip images in a given folder to AOI.
    
    Basic example: clip_images_to_aoi.py -in input_dir/images -out output_dir/images -shp input_dir/shapefile.shp
    see all other argument options for more details. 
    ''',formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-in', '--input_dir', type=dir_path, required=True, help=r'Path to directory containing geotiff files')
    parser.add_argument('-out', '--output_dir', type=str, required=True, help=r'Path to directory to write geotiff files')
    parser.add_argument('-shp', '--shape_file', type=str, required=True, help=r'AOI shape file')
    parser.add_argument('-debug', '--debug', action='store_true', required=False, help='Set this flag for debug output (outputs unmerged polygons)')
    args = parser.parse_args()

    log.info(f'Input geotiffs directory: {args.input_dir}')
    tifs_dir_path = Path(args.input_dir)
    assert tifs_dir_path.is_dir(), f"Input directory {str(tifs_dir_path)} is not readable"
    gpaths = list(tifs_dir_path.glob("**/*.tif"))
    num_files = len(gpaths)
    log.info(f"Number of mask files to process: {num_files}")
    if num_files <1:
        log.info("No files to process: terminating")
        return

    log.info(f'AOI shape file: {args.shape_file}')
    assert os.path.exists(args.shape_file), f"AOI shape file {args.shape_file} is not readable"
    
    log.info(f'Output geotiffs directory: {args.output_dir}')
    tifs_output_path = Path(args.output_dir)
    # if output directory doesn't exist, create it
    tifs_output_path.parent.mkdir(exist_ok=True, parents=True)


     # set logging level
    if args.debug:
        log.setLevel(logging.DEBUG)
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)
        log.addHandler(ch)
    else:
        log.setLevel(logging.INFO)
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        log.addHandler(ch)

    log.info(f'Clipping geotiffs to AOI shape file...')
        
    with fiona.open(args.shape_file, "r") as shapefile:
        shapes = [feature["geometry"] for feature in shapefile]
    
    
        for geotiff_path in tqdm(gpaths):
            image_id = geotiff_path.name
    
            with rasterio.open(str(geotiff_path)) as src:
                try:
                    out_image, out_transform = rasterio.mask.mask(src, shapes, crop=True)
                    out_meta = src.meta    
                
                    out_meta.update({"driver": "GTiff",
                        "height": out_image.shape[1],
                        "width": out_image.shape[2],
                        "transform": out_transform})
                
                    output_file = args.output_dir + "/" +image_id
                    log.info(f'Output file: {output_file}')
                    with rasterio.open(output_file, "w", **out_meta, compress="JPEG", photometric="YCbCr") as dest:
                        dest.write(out_image)
                except ValueError:
                    print(f'{image_id} is outside AOI.  Skipping image...')

if __name__=="__main__":
    main()
