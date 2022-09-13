#################################################################
#
# created for ramp project, August 2022
# Author: carolyn.johnston@dev.global
#
#################################################################

 
 
import os, sys
from pathlib import Path
import rasterio as rio
import numpy as np
import argparse
import tensorflow as tf
import solaris as sol
from tqdm import tqdm

# adding logging
# options for log levels: INFO, WARNING, DEBUG
import logging
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
log.addHandler(ch)

from ramp.utils.label_utils import create_tile_gdf

# get rid of geopandas FutureWarning
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

def main():

    parser = argparse.ArgumentParser(description='''
    Generate tiles for training from an image and label set,
    or generate tiles for production from an image.
    The image must first have been standardized for use in ramp processing: 
    it should be a 3-channel unsigned integer RGB geotiff with 0 as its nodata value. 

    Example: tile_datasets.py -img large_AOI_image.tif -vecs large_AOI_vectors.geojson -out output_dir
        -pfx large_AOI -trec directory_of_tiles.geojson -ndt 0.4 -oht 256 -owd 256

        will produce image chips in the directory output_dir/chips, and labels in the directory output_dir/labels.
        These will have names of the form large_AOI_*.tif and large_AOI*.geojson.
        directory_of_chips.geojson is a record of the names and locations of created chips.
        Image chips containing more than 40% invalid data will be skipped and not created.
    ''', 
    formatter_class=argparse.RawTextHelpFormatter)

    # required
    parser.add_argument('-img', '--image_file', type=str, required=True, help=r'Path to input image.')
    parser.add_argument('-vecs', '--vectors_file', type=str, required=False, default=None, help='Path to input labels (optional)')
    parser.add_argument('-out', '--output_dir', type=str, required=True, help="Path to output directory")
    parser.add_argument('-pfx', '--prefix', type=str, required=True, help="prefix for chip and label filenames")

    # optional/has default
    parser.add_argument('-trec', '--tile_record_filename', type=str, required=False, default=None, help="geojson database of chips created")
    parser.add_argument('-ndt', '--nodata_threshold', type=float, required=False, default=0.4, help="threshold for discarding images with a higher percentage of nodata values")
    parser.add_argument('-oht', '--output_tile_height', type=int, required=False, default=256, help="height in pixels of output tiles")
    parser.add_argument('-owd', '--output_tile_width', type=int, required=False, default=256, help="width in pixels of output tiles")
    args = parser.parse_args()

    img_file = args.image_file
    assert Path(img_file).is_file(), f"Image file {img_file} is not readable"
    
    # get image information
    raster = sol.utils.core._check_rasterio_im_load(img_file)
    # src_img_shape = raster.shape
    src_img_nodata = 0
    
    labels_file = args.vectors_file
    tile_labels = False
    if labels_file is not None:
        tile_labels = True
        assert Path(labels_file).is_file(), f"Vectors file {labels_file} is not readable"

    # create output directories for chips
    out_dir = args.output_dir
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    chip_dir = str(Path(out_dir)/"chips") 
    Path(chip_dir).mkdir(parents=True, exist_ok=True)

    # create output directories for labels
    if tile_labels:
        label_dir = str(Path(out_dir)/"labels")
        Path(label_dir).mkdir(parents=True, exist_ok=True)
    
    file_prefix = args.prefix
    nodata_thresh = args.nodata_threshold
    tile_record_filename = args.tile_record_filename
    create_tile_record = False if tile_record_filename is None else True

    dst_shape=(args.output_tile_height, args.output_tile_width)

    # src_shape means the size of tile to cut out from the source tile when a tile is made. 
    # this should be the same size as the destination shape to prevent downsampling.
    src_shape=dst_shape
    
    log.info(f"Tiling image file: {img_file}")

    raster_tiler = sol.tile.raster_tile.RasterTiler(
        dest_dir = chip_dir,
        dest_tile_size = dst_shape,
        src_tile_size = src_shape,
        verbose=True,
        alpha=None
    )

    raster_bound_crs = raster_tiler.tile(
        img_file,
        dest_fname_base = file_prefix,
        nodata_threshold=nodata_thresh
    )

    if tile_labels:
        log.info(f"Tiling vector file: {labels_file}")
        vector_tiler = sol.tile.vector_tile.VectorTiler(
            dest_dir = label_dir,
            verbose=True
        )

        vector_tiler.tile(
            labels_file,
            tile_bounds = raster_tiler.tile_bounds,
            tile_bounds_crs = raster_bound_crs,
            dest_fname_base = file_prefix
        )

    if create_tile_record:
        log.info(f"Creating tile record file: {tile_record_filename}")
        gdf = create_tile_gdf(
            raster_tiler = raster_tiler, 
            tile_bounds_crs=raster_bound_crs
            )
        gdf.to_file(tile_record_filename, driver="GeoJSON")

if __name__=="__main__":
    main()