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

import rasterio as rio
from rasterio.warp import calculate_default_transform, reproject, Resampling
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
    Create fused geojson polygon outputs from a directory of multichannel masks.
    Use with truth masks to get fused truth building data, and with predicted masks to 
    get fused building predictions.
    
    Basic example: get_labels_from_masks.py -in multimasks -out fused_buildings.geojson
    see all other argument options for more details. 
    ''',formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-in', '--input_dir', type=dir_path, required=True, help=r'Path to directory containing input multichannel masks.')
    parser.add_argument('-out', '--out_json', type=str, required=True, help=r'Path to output label file (geojson)')
    parser.add_argument('-bin', '--binary_mask', action='store_true', required=False, help='Set this flag if masks are binary (default: 4-channel masks)')
    parser.add_argument('-bwpix', '--bdry_width_pixels', type=int, required=False, default=2, help='''
                        The width of the building boundaries in the multichannel masks used in training. 
                        Default: 2. 
                        ''')
    parser.add_argument("-tmpdir", "--temp_mask_dir", type=str, required=False, default=None, help="temporary directory for writing reprojected masks")
    parser.add_argument("-crs", "--out_crs", type=str, required=False, default="EPSG:4326", help="EPSG code for output label file CRS (optional: default EPSG:4326)" )
    parser.add_argument('-debug', '--debug', action='store_true', required=False, help='Set this flag for debug output (outputs unmerged polygons)')
    args = parser.parse_args()

    log.info(f'Input masks directory: {args.input_dir}')
    mask_dir_path = Path(args.input_dir)
    assert mask_dir_path.is_dir(), f"Mask directory {str(mask_dir_path)} is not readable"
    mpaths = list(mask_dir_path.glob("**/*.tif"))
    num_files = len(mpaths)
    log.info(f"Number of mask files to process: {num_files}")
    if num_files <1:
        log.info("No files to process: terminating")
        return

    log.info(f'Output labels file: {args.out_json}')
    output_filepath = Path(args.out_json)

    # if directory doesn't exist, create it
    output_filepath.parent.mkdir(exist_ok=True, parents=True)

    if args.binary_mask:
        log.info(f'Processing binary masks')
    else:
        log.info(f'Processing multichannel masks')

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

    buffer_pixels = args.bdry_width_pixels*2+1
    
    # loop over mask files

    affine_object = None
    building_dfs = []

    ## As part of processing, we will create a temporary directory
    # for reprojected masks.
    # either use an input temporary directory, or use the default.
    if args.temp_mask_dir is None:
        
        # use default temp dir location
        parent_path = mask_dir_path.parent
        crs_4326_mask_path = parent_path/"crs_4326"
    else:
        # use requested temp dir location
        crs_4326_mask_path = Path(args.temp_mask_dir)

    crs_4326_mask_path.mkdir(parents=True, exist_ok=True)
    log.info(f"Storing reprojected masks in directory: {str(crs_4326_mask_path)}")

    for mask_path in tqdm(mpaths):

        image_id = mask_path.name
        

        ####### 20220825 CJ: START reprojecting masks to EPSG:4326
        with rio.open(str(mask_path)) as mask_src:

            # calculate transform data
            affine_object, width, height = calculate_default_transform(
                                mask_src.crs, 
                                'EPSG:4326', 
                                mask_src.width, 
                                mask_src.height,
                                *mask_src.bounds
                                )

            # get profile for reprojected mask
            kwargs = mask_src.meta.copy()
            kwargs.update({
                'crs':'EPSG:4326',
                'transform':affine_object,
                'width':width,
                'height':height
            })

            # reproject mask and write to file
            mask_name = mask_path.name
            reproject_mask_path = crs_4326_mask_path/mask_name            
            with rio.open(str(reproject_mask_path), mode='w', **kwargs) as mask_dst:
                reproject(
                    source=rio.band(mask_src,1),
                    destination=rio.band(mask_dst,1),
                    src_transform=mask_src.transform,
                    src_crs=mask_src.crs,
                    dst_transform=affine_object,
                    dst_crs="EPSG:4326",
                    resampling=Resampling.nearest
                )

        # read reprojected mask from file
        with rio.open(str(reproject_mask_path)) as mask_src:
            transform = mask_src.profile["transform"]
            mask_array = mask_src.read(1)

        ####### 20220825 CJ: END reprojecting masks to EPSG:4326

        # if (sparse) multichannel mask, convert to a binary mask
        binmask = binary_mask_from_multichannel_mask(mask_array) if not args.binary_mask else mask_array

        # get geodataframe 
        building_df = binary_mask_to_geodataframe(
                    binmask, 
                    affine_object=transform,
                    df_crs="EPSG:4326", 
                    do_transform=True, # we want output in degrees, not pixels
                    min_area=0 # set this to 0 to avoid problems with coordinate units
        )

        # add the mask filename to the geodataframe so it's associated with every building polygon
        # append nonempty geodataframes to the list
        num_bldgs = len(building_df)
        log.debug(f"Buildings in {image_id}: {num_bldgs}")
        if num_bldgs != 0:

            file_list = [image_id]*num_bldgs
            building_df["image_id"] = file_list
            building_dfs.append(building_df)

    # COMMENT: now all building dfs are in WGS84 lat-lon. 
    # We assume that the linear portion of ANY affine transform associated with a mask gives a roughly accurate
    # linear portion of an affine transform for any of the masks.
    # this is equivalent to assuming that the extent of the whole AOI is not so large that the distance between two 
    # longitudes changes a lot from the northern to the southern extreme of the AOI.
    # If this is not the case, you should break up production into datasets with smaller north-south extents.
    num_dfs = len(building_dfs)
    log.info(f"Number of nonempty dataframes: {num_dfs}")
    if num_dfs <1:
        log.info("No buildings were extracted: terminating")
        return

    full_geodf = pd.concat(building_dfs, axis=0, ignore_index=True)
    full_geodf["polyid"] = range(len(full_geodf))

    if args.debug:
        outfile = str(output_filepath).replace(".geojson", "_unbuffered.geojson")
        log.debug(f"output tile polygon set to: {outfile}")
        full_geodf.to_file(outfile)
    
    ### Begin polygon post-processing

    # buffer polys by x pixels
    buff_geodf = buffer_df_geoms(full_geodf, buffer_pixels, affine_obj=affine_object)
        
    if args.debug:
        outfile = str(output_filepath).replace(".geojson", "_buffered.geojson")
        log.debug(f"output buffered tiled polygon set to: {outfile}")
        buff_geodf.to_file(outfile)

    # get pairwise intersections of all polygons in all tiles
    df_join = full_geodf.sjoin(buff_geodf, how="left")

    # filter out intersections among buildings in the same tile (includes self-intersections)
    df_filtered = df_join[df_join["image_id_left"]!=df_join["image_id_right"]]
    if args.debug and len(df_filtered)>0:
        outfile = str(output_filepath).replace(".geojson", "_joined.geojson")
        log.debug(f"output spatial join polygon set to: {outfile}")
        df_filtered.to_file(outfile)

    # get equivalence classes (by index) of polygons in different tiles that should be merged
    # returns full_geodf with an additional column identifying groups of polygons to merge
    full_geodf = get_polygon_indices_to_merge(full_geodf, df_filtered)

    if args.debug and len(full_geodf)>0:
        unmerged_output_json = str(output_filepath).replace(".geojson", "_with_merge_class.geojson")
        log.debug(f"Writing polygons with merge class to file: {unmerged_output_json}")
        full_geodf.to_file(unmerged_output_json)

    # merge across tiles by dissolving buffered polygons in the same merge class
    # then 'unbuffering' the dissolve results.    
    buff_geodf["merge_class"] = full_geodf["merge_class"]
    merged_buffered_geodf = buff_geodf.dissolve(by="merge_class")
    merged_geodf = buffer_df_geoms(merged_buffered_geodf, -buffer_pixels, affine_obj=affine_object)
        
    if args.debug and len(merged_geodf)>0:
        unfilled_output_json = str(output_filepath).replace(".geojson", "_unfilled.geojson")
        log.debug(f"Writing (unfilled) polygons to file: {unfilled_output_json}")
        merged_geodf.to_file(unfilled_output_json)

    # eliminate rings in the merged polygons
    merged_geodf["geometry"] = merged_geodf.geometry.apply(lambda p: fill_rings(p))

    # finally, buffer again to recover the area of the building boundaries,
    # this time by the boundary width of the masks used in training
    merged_geodf = buffer_df_geoms(merged_geodf, args.bdry_width_pixels, affine_obj=affine_object)

    ## add area (m2) and perimeter (m) attributes to the final output
    merged_geodf = gpd_add_area_perim(merged_geodf)

    if args.debug and len(merged_geodf)>0:
        filled_output_json = str(output_filepath).replace(".geojson", "_norings_area.geojson")
        log.debug(f"Writing filled polygons to file: {filled_output_json}")
        merged_geodf.to_file(filled_output_json)

    if len(merged_geodf)>0:

        # CJ 20220825: reproject to specified output CRS
        if args.out_crs == "EPSG:4326":
            reprojected_merged_geodf = merged_geodf
        else:
            reprojected_merged_geodf = merged_geodf.to_crs(args.out_crs)

        log.info(f"Writing label polygons to file: {str(output_filepath)}")
        reprojected_merged_geodf.to_file(str(output_filepath),driver="GeoJSON")
    else:
        log.info("Output dataframe is empty: no file written")
    return

if __name__=="__main__":
    main()
