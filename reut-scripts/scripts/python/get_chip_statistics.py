#################################################################
#
# created for ramp project, August 2022
# Author: carolyn.johnston@dev.global
#
#################################################################

 
 
# adding logging
# some options for log levels: INFO, WARNING, DEBUG
import logging
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
log.addHandler(ch)

# get rid of geopandas FutureWarning
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import sys, os
import numpy as np
import argparse
from pathlib import Path
import pandas as pd
import geopandas as gpd
import rasterio as rio
from rasterio import features
from shapely.geometry import box
from tqdm import tqdm

RAMP_HOME = os.environ["RAMP_HOME"]


from ramp.utils.file_utils import get_matching_pairs, get_basename
from ramp.utils.img_utils import gdal_get_image_tensor, nodata_count, get_nodata_mask_from_image, to_channels_last, to_channels_first
from ramp.utils.label_utils import add_overlap_validpoly_columns

def main():
    parser = argparse.ArgumentParser(description='''
    Given a training data set consisting of directories of 
    images and truth label geojsons with matching basenames,
    calculate statistics such as the number of buildings, 
    maximum and minimum building areas, 
    and whether buildings are defined over nodata values.

    Creates a csv file with per-chip/per-label statistics. 
    Example: get_chip_statistics.py -idir chips -ldir labels -csv chip_statistics.csv
    
    Also creates a geojson file with the chip boundary polygon included, 
    suitable for loading into QGIS,
    having the same basename as the csv file (chip_statistics.geojson).

    Building polygons defined over nodata values must be repaired so that they are not defined over nodata values. In order to help find buildings defined over nodata values, 
    use the 'save_bad_pixel_dir' switch. Geotiff masks showing the locations of nodata values under buildings
    will be saved to this directory.
    ''',
    formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('-idir', '--image_dir', type = str, required = True, help="Directory containing image chip geotiffs")
    parser.add_argument('-ldir', '--label_dir', type =str, required = True,  help='Directory containing label geojsons')
    parser.add_argument('-csv', '--csv', type=str, required=True, help='path to csv file of chip statistics' )
    parser.add_argument('-bpdir', '--save_bad_pixel_dir', type=str, required=False, default=None, 
                help="Directory to save bad-pixel masks (of nodata values under buildings) to (default: don't save)" )
    args = parser.parse_args()

    imdir = Path(args.image_dir)
    assert imdir.is_dir(), f"image directory {str(imdir)} not readable"
    ldir = Path(args.label_dir)
    assert ldir.is_dir(), f"label directory {str(ldir)} not readable"
    
    outcsv = Path(args.csv)
    
    # added: create geojson output with chip bounds
    pdir = outcsv.parent
    basename = outcsv.stem
    geojson_file = pdir / (basename + '.geojson')

    bad_pix_dir = args.save_bad_pixel_dir

    if bad_pix_dir is not None:
        bad_pix_dir = Path(bad_pix_dir)
        bad_pix_dir.mkdir(parents=True, exist_ok=True)
    
    imfiles = imdir.glob("**/*.tif")
    lfiles = ldir.glob("**/*.geojson")
    matched_pairs = get_matching_pairs(imfiles, lfiles)

    # lists that will be populated for final chip stats output
    fprint_counts = []
    nodata_counts = []
    nodata_bldg_pixels_counts = []
    imfiles = []
    lfiles = []
    geoms = []
    overlap_counts = []
    invalid_counts = []

    for (imfile, lfile) in tqdm(matched_pairs):

        # add filenames to output lists
        imfiles.append(str(imfile))
        lfiles.append(str(lfile))

        # process each image-label pair
        with rio.open(imfile) as src_img:
            affine = src_img.transform
            the_tensor = to_channels_last(src_img.read())
            crs = src_img.crs


            # get a polygon describing the chip bounds
            bounds = src_img.bounds
            geom = box(*bounds)
            geoms.append(geom)

            its_nodata_count = nodata_count(the_tensor)
            nodata_counts.append(its_nodata_count)

            # read label file for number of labels
            labels = gpd.read_file(lfile)
            its_fp_count = len(labels)
            fprint_counts.append(its_fp_count)

            # detect invalid polygons, and number of pairwise overlaps with other labels
            labels = add_overlap_validpoly_columns(labels)
            # log.info(f"columns in labels: {labels.columns}")
            invalid_counts.append(its_fp_count - sum(list(labels['is_valid'])))
            overlap_counts.append(sum(list(labels["overlaps"])))


            if its_nodata_count !=0 and its_fp_count !=0:

                # todo: check for nodata under buildings!
                # step 1: do you need to reproject the labels?
                if labels.crs.to_epsg() != src_img.crs.to_epsg():
                    labels = labels.to_crs(src_img.crs)

                # get the mask of nodata pixels
                nodata_mask = get_nodata_mask_from_image(the_tensor)
                
                # get the building mask
                label_mask = features.rasterize(
                    labels.geometry,
                    out_shape = src_img.shape,
                    fill = 0,
                    out = None,
                    transform = src_img.transform,
                    all_touched = False,
                    default_value = 1,
                    dtype = np.uint8)

                # get the intersection of nodata and buildings masks
                # get the number of pixels in the intersection
                nodata_bldgs_mask = np.logical_and(nodata_mask, label_mask)
                nodata_bldgs_count = np.sum(nodata_bldgs_mask)
                nodata_bldg_pixels_counts.append(nodata_bldgs_count)

                # if there are 'bad pixels' and you are saving bad pixel masks:
                if nodata_bldgs_count != 0 and bad_pix_dir is not None:

                    # save bad pixel mask to bad_pix_dir
                    profile = src_img.profile
                    profile.update(
                        dtype=rio.uint8,
                        count=1
                    )
                    basename = get_basename(imfile)
                    badmask_filename = bad_pix_dir / (basename + '.mask.tif')
                    with rio.open(badmask_filename, 'w', **profile) as dst:
                        dst.write(np.expand_dims(nodata_bldgs_mask,axis = 0))
            else:
                nodata_bldg_pixels_counts.append(0)

    # Done looping over chip-label pairs: write the statistics to CSV

    stats = pd.DataFrame(
        {
            "image_file":imfiles,
            "label_file":lfiles,
            "nodata_count":nodata_counts,
            "num_bldgs":fprint_counts,
            "num_nodata_under_bldg":nodata_bldg_pixels_counts,
            "num_invalid_polys":invalid_counts,
            "num_label_overlaps":overlap_counts
        }
    )
    log.info(f"Writing csv file: {str(outcsv)}")
    stats.to_csv(outcsv)

    geostats = gpd.GeoDataFrame(
        {
            "image_file":imfiles,
            "label_file":lfiles,
            "nodata_count":nodata_counts,
            "num_bldgs":fprint_counts,
            "num_nodata_under_bldg":nodata_bldg_pixels_counts,
            "num_invalid_polys":invalid_counts,
            "num_label_overlaps":overlap_counts,
            "geometry":geoms
        }
    )

    # bugfix 20220831: set chipstat file crs
    geostats=geostats.set_crs(crs)
    log.info(f"Writing geojson file: {geojson_file}")
    geostats.to_file(geojson_file)

if __name__=="__main__":
    main()