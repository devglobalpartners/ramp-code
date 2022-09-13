#################################################################
#
# created for ramp project, August 2022
# Author: carolyn.johnston@dev.global
#
#################################################################

 
 
from itertools import count
import os, argparse, sys, shutil
from pathlib import Path
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from tqdm import tqdm
import geopandas as gpd
from ramp.utils.geo_utils import gpd_add_area_perim
import csv

# adding logging
# options for log levels: INFO, WARNING, DEBUG
import logging
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
log.addHandler(ch)


def check_if_filterable(in_path):
    '''
    filterable means non empty after removal of null geometry
    returns the validated dataframe (vdf) if it's filterable, returns None otherwise
    '''
    vdf = gpd.read_file(in_path)
    if len(vdf) == 0:
        log.warning(f"Empty label dataframe: {in_path}")
        return None

    # remove NULL geometries
    valid = [item is not None for item in vdf["geometry"]]
    vdf = vdf[valid]
    before_count = len(vdf)
    if before_count == 0:
        log.warning(f"Empty label dataframe: {in_path}")
        return None

    return vdf

def filter_label_file_by_size(min_size, in_path, out_path):
    '''
    Returns 2-tuple: (number of buildings before filtering, number of buildings after filtering)
    '''
    out_path_dir = Path(out_path).parent

    if min_size <=0:
        return

    vdf = check_if_filterable(in_path)
    if vdf is None:
        # move file to new directory and return
        shutil.copy(in_path, str(out_path_dir))
        return

    before_count = len(vdf)

    # we need to do this conversion in order to use the geodetic area functions
    vdf = vdf.to_crs(4326)
    vdf = gpd_add_area_perim(vdf)
    vdf_filtered = vdf.query(f'area_m2 >= {min_size}')
    after_count = len(vdf_filtered)
    if after_count == 0:
        log.warning(f"Empty label dataframe after filtering: {in_path}")
        shutil.copy(in_path, str(out_path_dir))
        return
    vdf_filtered.to_file(out_path, driver="GeoJSON")
    return (before_count, after_count)

def getCountInformation(labelfile, before_count, after_count):
    return {'filename':labelfile, 'before':before_count, 'after':after_count}


def main():

    parser = argparse.ArgumentParser(
        description='''
        Remove buildings with area (in sq m) smaller than a threshold from all geojson label files in a directory,
        and write the filtered geojson files to another directory.

        Example: remove_slivers.py -ldir input_labels_dir -fdir output_labels_dir -min 10.0
        
        ''',formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-ldir', '--label_file_dir', type =str, required = True, help='Path to directory containing input label files')
    parser.add_argument('-fdir', '--filtered_file_dir', type=str, required = False, default=None, help='Name of output directory for filtered label files')
    parser.add_argument('-min', '--min_sqmeters', type=float, required=True, help="minimum size buildings to keep, in sq meters")
    parser.add_argument('-csv', '--csv_fn', type=str, required=False, default=None, help='path to optional csv file for before and after building counts' )
    args = parser.parse_args()

    label_path = Path(args.label_file_dir)
    assert label_path.is_dir(), f"input directory {str(label_path)} is not readable"
    min_size = args.min_sqmeters
    output_dir = args.filtered_file_dir
    if output_dir is None:
        output_path = label_path.parent / f"filtered_labels_{min_size}"
        output_path.mkdir(exist_ok=True, parents=True)
    else:
        output_path = Path(output_dir) 
        output_path.mkdir(parents=True, exist_ok=True)
    log.info(f"Writing filtered label files to folder: {str(output_path)}")   

    csv_file = None if args.csv_fn is None else str(args.csv_fn)

    filter_info = []
    if min_size <=0:
        log.warning('Minimum size 0: nothing to do: returning')
        return

    log.info(f"Processing labels in directory: {str(label_path)}")

    lfiles = [fn for fn in label_path.glob('**/*') if fn.is_file()]

    for lf in tqdm(lfiles):
        # construct output file name.
        # We're keeping the original name so it will continue to match the image chip,
        # but we are storing it in a new directory.
        outfile = output_path / f"{lf.stem}.geojson"
        count_data = filter_label_file_by_size(min_size, str(lf), str(outfile))
        if count_data is not None:
            filter_info.append(getCountInformation(lf, count_data[0], count_data[1]))

    # write the csv file
    if csv_file:

        # if it exists, remove and rewrite
        if os.path.exists(csv_file):
            os.remove(csv_file)

        # if no filter info, return
        if len(filter_info) == 0:
            return
        
        # write the csv file
        keys = ['filename','before','after']
        with open(csv_file, "w") as fileh:
            dict_writer = csv.DictWriter(fileh, keys)
            dict_writer.writeheader()
            dict_writer.writerows(filter_info)
        
    return

if __name__ == "__main__":
    main()