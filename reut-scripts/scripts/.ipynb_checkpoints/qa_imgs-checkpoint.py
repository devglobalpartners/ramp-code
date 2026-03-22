#!/usr/bin/env python3
import os
import shutil
import time
import argparse

import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling


############################################################
#  HELPERS
############################################################

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


############################################################
#  RENAME TO .tif
############################################################

def rename_to_tif(file_path):
    """Renames file to .tif and returns new path."""
    directory, filename = os.path.split(file_path)

    if filename.lower().endswith(".tif"):
        return file_path  # nothing to change

    name, ext = os.path.splitext(filename)
    new_filename = name + ".tif"
    new_path = os.path.join(directory, new_filename)

    print(f"[Rename] {filename} → {new_filename}")
    os.rename(file_path, new_path)

    return new_path


############################################################
#  REMOVE NIR BAND
############################################################

def remove_nir(input_path, output_dir):
    """If TIFF has 4+ bands, rreturn the first three bands. Returns new file path. Assumes that the image has 4 bands : R,G,B,NIR, if case is differeent be careful!"""
    ensure_dir(output_dir)

    filename = os.path.basename(input_path)
    output_path = os.path.join(output_dir, filename)

    with rasterio.open(input_path) as src:
        if src.count < 4:
            print(f"[NIR-Skip] {filename}: only {src.count} bands")
            return input_path  # no change

        print(f"[NIR-Remove] {filename}")

        rgb_data = src.read([1, 2, 3])
        profile = src.profile
        profile.update(count=3)

        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(rgb_data)

    return output_path


############################################################
#  REPROJECT
############################################################

def reproject_if_needed(input_path, output_dir, expected_crs):
    """Checks CRS and reprojects only if needed."""
    ensure_dir(output_dir)

    filename = os.path.basename(input_path)
    output_path = os.path.join(output_dir, filename)

    with rasterio.open(input_path) as src:
        current_crs = src.crs.to_string() if src.crs else None

        if current_crs == expected_crs:
            print(f"[CRS-OK] {filename} already in {expected_crs}")
            return input_path

        print(f"[Reproject] {filename}: {current_crs} → {expected_crs}")

        transform, width, height = calculate_default_transform(
            src.crs, expected_crs, src.width, src.height, *src.bounds
        )

        kwargs = src.meta.copy()
        kwargs.update({
            "crs": expected_crs,
            "transform": transform,
            "width": width,
            "height": height,
        })

        with rasterio.open(output_path, "w", **kwargs) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_crs=src.crs,
                    dst_crs=expected_crs,
                    resampling=Resampling.nearest,
                )

    return output_path


############################################################
#  MAIN PIPELINE
############################################################

def preprocess_all(input_dir, output_dir, expected_crs="EPSG:4326"):
    ensure_dir(output_dir)

    for filename in os.listdir(input_dir):
        if filename.startswith("."):
            continue

        print(f"\n=== Processing: {filename} ===")

        current_path = os.path.join(input_dir, filename)
        name, ext = os.path.splitext(filename)

        # # 1) Fix suffix
        # if ext.lower() != ".tif":
        #     print("[Suffix] Fixing suffix...")
        #     current_path = rename_to_tif(current_path)
        #     filename = os.path.basename(current_path)

        
        # 1) Fix suffix — always force .tif extension on disk
        if ext.lower() != ".tif":
            print("[Suffix] Fixing suffix...")
            new_path = rename_to_tif(current_path)   # this modifies the file on disk
            current_path = new_path                  # update the working path
            filename = os.path.basename(new_path)
        # Open raster info
        with rasterio.open(current_path) as src:
            band_count = src.count
            crs = src.crs.to_string() if src.crs else None

        # 2) Remove NIR
        if band_count > 3:
            current_path = remove_nir(current_path, output_dir)

        # 3) Reproject if needed
        with rasterio.open(current_path) as src:
            crs = src.crs.to_string() if src.crs else None

        if crs != expected_crs:
            current_path = reproject_if_needed(
                input_path=current_path,
                output_dir=output_dir,
                expected_crs=expected_crs,
            )

        # 4) Copy final file into output dir
        final_path = os.path.join(output_dir, filename)
        if current_path != final_path:
            shutil.copy2(current_path, final_path)

        print(f"[Done] Saved → {final_path}")


############################################################
#  ARGPARSE WRAPPER
############################################################

def main():
    parser = argparse.ArgumentParser(
        description="Raster preprocessing pipeline: rename → remove NIR → reproject."
    )

    parser.add_argument("--input_dir", required=True, help="Directory with input rasters")
    parser.add_argument("--output_dir", required=True, help="Directory to store results")
    parser.add_argument("--expected_crs", default="EPSG:4326", help="Target CRS")

    args = parser.parse_args()

    preprocess_all(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        expected_crs=args.expected_crs
    )


if __name__ == "__main__":
    main()
