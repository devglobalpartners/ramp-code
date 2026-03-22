#!/usr/bin/env python3
"""
raster_intersect.py

Usage example (one-line):
python3 raster_intersect.py --rasters_path "/home/jovyan/work/imagery/Nepal/merged_04122025/imgs_bgr_to_rgb" \
    --polygon_path "/home/jovyan/work/imagery/Nepal/merged_04122025_clean.gpkg" \
    --dest_raster_path "/home/jovyan/work/imagery/Nepal/merged_04122025/rgb_clean"
"""

import argparse
import os
import glob
import shutil
import pathlib
import warnings

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.mask import mask
from shapely.geometry import box
from tqdm import tqdm
import matplotlib.pyplot as plt

# CONSTANTS
GEOM_COL_NAME = 'geometry'
PATH_COL_NAME = 'path'
SUFFIX_FOR_TIF_FOLDER = "/*.tif"


def save_raster(img: np.ndarray, number_of_bands: int, transf: tuple,
                paths: str, tiff_name: str, crs_img: rasterio.crs.CRS):
    savedir = pathlib.Path(paths)
    savedir.mkdir(parents=True, exist_ok=True)

    with rasterio.open(
        savedir / tiff_name,
        'w',
        driver='COG',
        height=img.shape[1],
        width=img.shape[2],
        count=number_of_bands,
        dtype=img.dtype,
        crs=crs_img,
        nodata=None,
        transform=transf
    ) as dst:
        for n in range(number_of_bands):
            dst.write(img[n, :, :], int(n + 1))


def get_list_of_rasters(raster_storage_path: str, folder_suffix: str = SUFFIX_FOR_TIF_FOLDER):
    str_for_glob = raster_storage_path + folder_suffix
    list_of_rasters = glob.glob(str_for_glob)
    if len(list_of_rasters) == 0:
        warnings.warn(f"No rasters were found in {raster_storage_path}", UserWarning)
    else:
        print(f"Found {len(list_of_rasters)} rasters in {raster_storage_path}")
    return list_of_rasters


def raster_extent_to_vector(raster_path: str):
    src = rasterio.open(raster_path)
    src_crs = src.crs
    raster_extent = src.bounds
    polygon_gdf = gpd.GeoDataFrame(geometry=[box(*raster_extent)])
    polygon_gdf = polygon_gdf.set_crs(src_crs)
    return polygon_gdf, src_crs


def get_raster_intersects_with_vector(rasters_path, polygon_path, dest_raster_path, show_plots=False):
    # Read polygon tiles
    print("Reading polygon/geopackage:", polygon_path)
    gdf = gpd.read_file(polygon_path)
    total_tiles = len(gdf)
    print(f"Total tiles in polygon file: {total_tiles}")

    # Ensure destination exists
    os.makedirs(dest_raster_path, exist_ok=True)

    # Build set of existing filenames in dest folder (by basename)
    existing_dest_files = {os.path.basename(p) for p in glob.glob(os.path.join(dest_raster_path, "*"))}
    print(f"Files already in destination folder: {len(existing_dest_files)}")

    # If PATH_COL_NAME not present, warn and attempt to match by basename later
    has_path_col = PATH_COL_NAME in gdf.columns
    if not has_path_col:
        print(f"Warning: '{PATH_COL_NAME}' column not found in polygon file. Will try to match rasters by basename inside rasters_path.")

    # Prepare list of rasters available in rasters_path (for fallback)
    raster_pool = get_list_of_rasters(rasters_path)

    # Determine which rows still need processing by checking destination filenames
    rows_to_process = []
    missing_path_count = 0
    for idx, row in gdf.iterrows():
        # Determine the expected source file and destination name
        if has_path_col and row[PATH_COL_NAME]:
            source_path = str(row[PATH_COL_NAME])
            dest_name = os.path.basename(source_path)
        else:
            # If no path column, try to build dest_name from an attribute or index (fallback: index.tif)
            dest_name = f"tile_{idx}.tif"
            source_path = None
            missing_path_count += 1

        dest_full = os.path.join(dest_raster_path, dest_name)
        if dest_name in existing_dest_files or os.path.exists(dest_full):
            # already processed
            continue
        else:
            # keep index and source info to process
            rows_to_process.append((idx, row, source_path, dest_name))

    print(f"Tiles missing in destination (to process): {len(rows_to_process)}")
    if missing_path_count:
        print(f"Note: {missing_path_count} tile(s) had no '{PATH_COL_NAME}' value and will attempt fallback matching or be saved as generated names.")

    # If nothing to do, exit early
    if len(rows_to_process) == 0:
        print("Nothing to process. Exiting.")
        return

    # Process tiles with tqdm
    count_copy = 0
    count_clip = 0
    count_failed = 0

    for idx, row, source_path, dest_name in tqdm(rows_to_process, desc="Processing tiles"):
        # Ensure we have a source file path
        file_path = source_path

        # If source path is not present or doesn't exist, attempt to find file by basename in raster_pool
        if not file_path or not os.path.exists(file_path):
            if source_path:
                # user supplied path but not found
                print(f"Source path from polygon not found on disk: {source_path}")
            # Attempt to find by basename in raster_pool
            candidates = [p for p in raster_pool if os.path.basename(p) == dest_name]
            if len(candidates) == 1:
                file_path = candidates[0]
                # print fallback info
                print(f"Found fallback raster for tile {idx}: {file_path}")
            elif len(candidates) > 1:
                file_path = candidates[0]
                print(f"Multiple candidates found for {dest_name}, using first: {file_path}")
            else:
                # no candidate found -> skip this tile
                print(f"Could not find source raster for tile {idx} (expected basename: {dest_name}). Skipping.")
                count_failed += 1
                continue

        # Double-check destination doesn't appear (race)
        dest_full = os.path.join(dest_raster_path, dest_name)
        if os.path.exists(dest_full):
            # another process may have created it; skip
            continue

        try:
            # Open raster
            raster = rasterio.open(file_path)

            # Compute extents as vector
            one_raster_to_vector, src_crs = raster_extent_to_vector(file_path)

            # Create a GeoDataFrame for the current tile geometry (single-row)
            temp_gdf = gpd.GeoDataFrame([row])
            # Ensure same CRS when computing areas / overlaps
            try:
                if temp_gdf.crs is None:
                    temp_gdf = temp_gdf.set_crs(one_raster_to_vector.crs)
                else:
                    temp_gdf = temp_gdf.to_crs(one_raster_to_vector.crs)
            except Exception:
                # if CRS mismatch, attempt to set to raster CRS
                temp_gdf = temp_gdf.set_crs(one_raster_to_vector.crs, allow_override=True)

            # Optionally show a plot (disabled by default)
            if show_plots:
                ax = one_raster_to_vector.plot(color="none", edgecolor="red", linewidth=2)
                temp_gdf.plot(ax=ax, color="blue", alpha=0.5)
                plt.show()

            # Compute IoU (intersection over union) in the same CRS
            union = temp_gdf.unary_union.union(one_raster_to_vector.unary_union)
            intersection = temp_gdf.unary_union.intersection(one_raster_to_vector.unary_union)
            intersection_area = intersection.area
            union_area = union.area
            iou = intersection_area / union_area if union_area != 0 else 0.0

            # If fully inside -> copy
            if iou == 1.0:
                shutil.copyfile(file_path, dest_full)
                count_copy += 1

            # If partial overlap -> clip & save
            elif 0.0 < iou < 1.0:
                masked_raster, transform = mask(raster, temp_gdf.geometry, invert=False)
                # we attempt to determine band count from masked_raster
                band_count = masked_raster.shape[0] if masked_raster.ndim == 3 else 1
                save_raster(
                    img=masked_raster,
                    number_of_bands=band_count,
                    transf=transform,
                    paths=dest_raster_path,
                    tiff_name=dest_name,
                    crs_img=raster.crs
                )
                count_clip += 1

            else:
                # no meaningful overlap
                # skip saving
                pass

        except Exception as e:
            print(f"Error processing tile idx={idx}, source={file_path}: {e}")
            count_failed += 1
            continue

    # Final summary
    print("\n=== Summary ===")
    print(f"Total tiles in polygon file:        {total_tiles}")
    print(f"Already existed in destination:     {len(existing_dest_files)}")
    print(f"Tiles attempted to process:         {len(rows_to_process)}")
    print(f"Copied (full overlap)              {count_copy}")
    print(f"Clipped (partial overlap)          {count_clip}")
    print(f"Failed / skipped (not found etc.)  {count_failed}")
    print(f"Total new outputs (copied+clipped) {count_copy + count_clip}")


def main():
    parser = argparse.ArgumentParser(description="Process rasters intersecting polygon tiles")
    parser.add_argument("--rasters_path", required=True, help="Folder with input rasters")
    parser.add_argument("--polygon_path", required=True, help="Path to polygon gpkg/geojson")
    parser.add_argument("--dest_raster_path", required=True, help="Output folder for processed rasters")
    parser.add_argument("--show_plots", action="store_true", help="Show plots while processing (slows down)")
    args = parser.parse_args()

    get_raster_intersects_with_vector(
        rasters_path=args.rasters_path,
        polygon_path=args.polygon_path,
        dest_raster_path=args.dest_raster_path,
        show_plots=args.show_plots
    )


if __name__ == "__main__":
    main()
