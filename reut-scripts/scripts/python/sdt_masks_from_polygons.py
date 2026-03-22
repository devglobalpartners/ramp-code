#################################################################
#
# created for ramp project, August 2022
# Author: carolyn.johnston@dev.global
#
#################################################################

 
 
import os, argparse, sys
from pathlib import Path

from ramp.utils.misc_ramp_utils import dir_path
from ramp.utils.img_utils import gdal_write_mask_tensor
from osgeo import gdal, ogr
from tqdm import tqdm
from ramp.data_mgmt.chip_label_pairs import get_tq_chip_label_pairs, construct_mask_filepath
from ramp.utils.sdt_mask_utils import ProcessLabelToSdtMask

# adding logging
# options for log levels: INFO, WARNING, DEBUG
import logging
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
log.addHandler(ch)

def main():

    parser = argparse.ArgumentParser(description=''' 
    Create matching truncated signed distance transform masks from a folder of geojson files.
    Requires the path to the matching image chips directory.

    Example: sdt_masks_from_polygons.py -in_vecs labels_dir -in_chips matching_chips_dir -out output_sdt_masks_dir
        -mpp meters_per_pixel_in_input_chips -usb 3 -lsb 3 

        will create signed distance transform masks with 7 pixel classes.

    NOTE: output masks will have values in the range [0,numclasses)
    i.e., 0...6 instead of -3...3
    ''', formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-in_vecs', '--input_vec_dir', type=dir_path, required=True, help=r'Path to directory containing geojson files.')
    parser.add_argument('-in_chips', '--input_chip_dir', type = dir_path, required = True, help='Path to directory containing image chip files with names matching geojson files.')
    
    # made this directory a string, since it might not exist yet. If not, it will be created
    parser.add_argument('-out', '--output_dir', type=str, required=True, help=r'Path to directory containing output SDT masks.')
    parser.add_argument('-mpp', '--m_per_pixel', default=0.3, type=float, help='meters per pixel of input image chip')
    parser.add_argument('-usb', '--upper_sdt_bound', default=3, type=int, help='upper bound of truncated signed distance transform')
    parser.add_argument('-lsb', '--lower_sdt_bound', default=-3, type=int, help='lower bound of truncated signed distance transform')
    
    args = parser.parse_args()

    in_poly_dir = args.input_vec_dir
    in_chip_dir = args.input_chip_dir
    out_mask_dir = args.output_dir
    mpp = float(args.m_per_pixel)
    ucb = int(args.upper_sdt_bound)
    lcb = int(args.lower_sdt_bound)
    
    log.info(f'Input polygons directory: {in_poly_dir}')
    log.info(f'Input matching image chips directory: {in_chip_dir}')
    log.info(f'Output SDT masks directory: {out_mask_dir}')
    log.info(f'Image chip meters per pixel: {mpp}')
    log.info(f'Mask upper bound: {ucb}')
    log.info(f'Mask lower bound: {lcb}')
    

    # If output mask directory doesn't exist, create it. 
    Path(out_mask_dir).mkdir(parents=True, exist_ok=True)
    
    # construct list of matching image chips from json paths
    # also check to be sure the matching image chips are present
    chip_label_pairs = None    
    chip_label_pairs = get_tq_chip_label_pairs(in_chip_dir, in_poly_dir)
    chip_paths, label_paths = list(zip(*chip_label_pairs))

    # construct the output mask file names from the json file names.
    mask_paths = [construct_mask_filepath(out_mask_dir, chip_path) for chip_path in chip_paths]

    # iterate through mask production for all chip/label pairs
    json_chip_mask_zips = zip(label_paths, chip_paths, mask_paths)
    for json_path, chip_path, mask_path in tqdm(json_chip_mask_zips):
        
        src_ds = gdal.Open(chip_path)
        sdt_array = ProcessLabelToSdtMask(json_path, src_ds, mpp, (lcb, ucb))

        # write out the output mask file
        # added 20220520: change output type to byte from signed int
        gdal_write_mask_tensor(mask_path, src_ds, sdt_array, gdal.GDT_Byte)

    return

if __name__=="__main__":
    main()