#################################################################
#
# created for ramp project, August 2022
# Author: carolyn.johnston@dev.global
#
#################################################################

 
 
 
import os, argparse, sys
from pathlib import Path
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from tqdm import tqdm
import geopandas as gpd
from ramp.utils.geo_utils import gpd_add_area_perim

# adding logging: options: WARNING, INFO, DEBUG
import logging
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
log.addHandler(ch)

def filter_label_file_by_size(min_size, in_path, out_path):
    
    if min_size <=0:
         return;
    vdf = gpd.read_file(in_path)
    if len(vdf) == 0:
        log.warning(f"Cannot write empty dataframe to file: {in_path}")
        return 
    # remove NULL geometries
    valid = [item is not None for item in vdf["geometry"]]
    vdf = vdf[valid]
    if len(vdf) == 0:
        log.warning(f"Cannot write empty dataframe to file: {in_path}")
        return

    # we need to do this conversion in order to use the geodetic area functions
    vdf = vdf.to_crs(4326)
    vdf = gpd_add_area_perim(vdf)
    vdf_filtered = vdf.query(f'area_m2 >= {min_size}')
    if len(vdf_filtered) == 0:
        log.warning(f"Cannot write empty dataframe to file: {in_path}")
        return
    vdf_filtered.to_file(out_path, driver="GeoJSON")
    return


def main():

    parser = argparse.ArgumentParser(description='''
                    Add area and perimeter attributes to a label file.

                    Example: add_area_to_labels.py -lfn lfilename.geojson -out lfilename_plus_area.geojson  
                    ''',formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-lfn', '--label_filename', type=str, required=True, help=r'Path to input geojson')
    parser.add_argument('-out', '--outfile', type=str, required=False, default=None, help='''Path to output geojson with area added (default: area.original_filename.geojson''')
    parser.add_argument('-crs', '--crs', type=str, required=False, default=None, help=r'CRS of label file (use only if label file has no crs)')
    args = parser.parse_args()

    label_file = args.label_filename
    # validate file
    label_path= Path(label_file)
    assert label_path.is_file(), f"Input file {label_file} is not readable"

    output_file = args.outfile
    if output_file is None:

        # create output file name for label file with added area and perimeter fields
        output_file = str(lf.parent / f"area.{lf.stem}.geojson")
    
    
    # set source CRS if it was given
    vdf = gpd.read_file(label_file)
    src_crs = args.crs 
    if src_crs is not None:
        vdf = vdf.set_crs(src_crs)

    # remove NULL and empty geometries
    valid = [item is not None for item in vdf["geometry"]]
    vdf = vdf[valid]
    vdf = vdf[~vdf.is_empty]
    
    # check for empty data frame
    if len(vdf) == 0:
        log.warning(f"Cannot write empty dataframe to file: {outfile}")
        return

    # to_crs converts the vdf to geo CRS. This is needed for use of the geodetic functions.
    vdf = vdf.to_crs(4326)
    vdf = gpd_add_area_perim(vdf)
    log.info(f"writing labels to {output_file}")
    vdf.to_file(output_file, driver="GeoJSON")
    return

if __name__ == "__main__":
    main()