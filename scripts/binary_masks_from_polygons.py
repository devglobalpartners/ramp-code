#################################################################
#
# created for ramp project, August 2022
# Author: carolyn.johnston@dev.global
#
#################################################################

 
 
import os, argparse, sys
from pathlib import Path

from ramp.utils.misc_ramp_utils import dir_path
from ramp.utils.img_utils import get_binary_mask_from_layer
from osgeo import gdal, ogr
from tqdm import tqdm
from ramp.data_mgmt.chip_label_pairs import get_tq_chip_label_pairs, construct_mask_filepath


# adding logging: options INFO, WARNING, DEBUG
import logging
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
log.addHandler(ch)

def main():

    parser = argparse.ArgumentParser(description='''
    Create binary building footprint masks from a folder of geojson files. 
    Also requires path to the matching image chips directory.

    Example: binary_masks_from_polygons -in_vecs vector_dir -in_chips chip_dir -out binary_mask_dir
    ''',formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-in_vecs', '--input_vec_dir', type=dir_path, required=True, help=r'Path to directory containing geojson files.')
    parser.add_argument('-in_chips', '--input_chip_dir', type = dir_path, required = True, help='Path to directory containing image chip files with names matching geojson files.')
    
    # made this directory a string, since it might not exist yet. If not, it will be created
    parser.add_argument('-out', '--output_dir', type=str, required=True, help=r'Path to directory containing output SDT masks.')
    args = parser.parse_args()

    in_poly_dir = args.input_vec_dir
    in_chip_dir = args.input_chip_dir
    out_mask_dir = args.output_dir

    log.info(f'Input polygons directory: {in_poly_dir}')
    log.info(f'Input matching image chips directory: {in_chip_dir}')
    log.info(f'Output binary masks directory: {out_mask_dir}')

    # If output mask directory doesn't exist, try to create it. 
    Path(out_mask_dir).mkdir(parents=True, exist_ok=True)
    
    # construct list of matching image chips from json paths
    # also check to be sure the matching image chips are present
    
    chip_label_pairs = get_tq_chip_label_pairs(in_chip_dir, in_poly_dir)
    chip_paths, label_paths = list(zip(*chip_label_pairs))

    # construct the output mask file names from the chip file names.
    # these will have the same base filenames as the chip files, 
    # with a mask.tif extension in place of the .tif extension.
    mask_paths = [construct_mask_filepath(out_mask_dir, chip_path) for chip_path in chip_paths]
    
    # construct a list of full paths to the mask files
    json_chip_mask_zips = zip(label_paths, chip_paths, mask_paths)

    for json_path, chip_path, mask_path in json_chip_mask_zips:
        
        # We will run this on very large directories, and some label files might fail to process.
        # We want to be able to resume mask creation from where we left off.         
        if Path(mask_path).is_file():
            continue

        log.info(f"processing JSON file: {json_path}")
        # open the input image chip file and json polygon files
        drv = ogr.GetDriverByName('GeoJSON')
        ogr_ds = drv.Open(json_path)
        lyr = ogr_ds.GetLayer()
        src_ds = gdal.Open(chip_path)

        # This function call will write out the mask array to the mask path.
        # the returned mask_array is not used in this context.
        mask_array = get_binary_mask_from_layer(lyr, src_ds, in_ram=False, mask_path=mask_path)
        

    return

if __name__=="__main__":
    main()