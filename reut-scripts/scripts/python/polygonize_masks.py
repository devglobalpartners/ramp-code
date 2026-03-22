#################################################################
#
# created for ramp project, August 2022
# Author: carolyn.johnston@dev.global
#
#################################################################

 
 
import os, argparse, sys
from pathlib import Path


from ramp.utils.misc_ramp_utils import dir_path
from ramp.utils.img_utils import gdal_get_mask_tensor
from ramp.utils.ramp_exceptions import GdalReadError
from osgeo import gdal
from tqdm import tqdm
from ramp.utils.mask_to_vec_utils import binary_mask_from_sdt_mask, binary_mask_to_geojson

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
    Create matching geojson polygon outputs from a directory of Signed-Distance-Transform-derived or binary masks.

    Example for polygonizing SDT masks: polygonize_masks.py -in SDT_masks_dir -out polygons_dir -sdt -t 3
    Example for polygonizing binary masks: polygonize_masks.py -in binary_masks_dir -out polygons_dir

    ''',formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-in', '--input_dir', type=dir_path, required=True, help=r'Path to directory containing input masks.')
    parser.add_argument('-out', '--output_dir', type=str, required=True, help=r'Path to directory containing output geojson files.')
    parser.add_argument('-sdt', '--sdt', action='store_true', required=False, help='Set this flag if polygonizing signed distance transform masks. Default is binary masks')
    parser.add_argument('-t', '--thresh', default=3, help=r'use for SDT masks: threshold value for in-building pixels (default:3)') 
    args = parser.parse_args()

    mask_is_binary = not bool(args.sdt)
    sdt_mask_thresh = int(args.thresh)
    print(f'Input masks directory: {args.input_dir}')
    print(f'Output polygons directory: {args.output_dir}')
    
    if mask_is_binary:
        print(f'Polygonizing BINARY masks')
    else:
        print(f'Polygonizing SDT masks')
        print(f'Threshold value for building pixels: {sdt_mask_thresh}')

    # if output directory doesn't exist, create it
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # construct list of json paths from mask paths
    mask_tifs = [mask_tif for mask_tif in os.listdir(args.input_dir) if mask_tif.endswith('.tif')]
    basenames = [os.path.splitext(mask_tif)[0] for mask_tif in mask_tifs]
    mask_jsons = [basename + '.geojson' for basename in basenames]
    json_paths = [os.path.join(args.output_dir, mask_json) for mask_json in mask_jsons]
    mask_paths = [os.path.join(args.input_dir, mask_tif) for mask_tif in mask_tifs]
    json_mask_pairs = zip(mask_paths, json_paths)
    
    for mask_path, json_path in tqdm(json_mask_pairs):
        ref_ds  = gdal.Open(mask_path)
        if not ref_ds:
            log.error(f"GDAL could not read {mask_path}")
            raise GdalReadError(f"Could not read {mask_path}")

        if mask_is_binary:
            bin_mask = gdal_get_mask_tensor(mask_path)
            binary_mask_to_geojson(bin_mask, ref_ds, json_path)
        else:
            # mask is an SDT mask, convert to a binary mask first
            sdt_arr = gdal_get_mask_tensor(mask_path)
            bin_mask = binary_mask_from_sdt_mask(sdt_arr, sdt_mask_thresh)
            binary_mask_to_geojson(bin_mask, ref_ds, json_path)
    
    return

if __name__=='__main__':
    main()