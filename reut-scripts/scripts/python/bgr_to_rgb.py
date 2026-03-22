#!/usr/bin/env python3

import os
import argparse
from concurrent.futures import ThreadPoolExecutor
import rasterio

def convert_bgr_to_rgb(filename, input_folder, output_folder):
    in_path = os.path.join(input_folder, filename)
    out_path = os.path.join(output_folder, filename)

    with rasterio.open(in_path) as src:
        img = src.read()  # (C, H, W)
        profile = src.profile

    rgb = img[[2, 1, 0], :, :]  # Swap BGR → RGB
    profile.update(count=3)

    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(rgb)

    print(f"Saved RGB file: {out_path}")

def main():
    parser = argparse.ArgumentParser(description="Convert BGR TIFF images to RGB.")
    parser.add_argument("-input", required=True, help="Input folder containing TIFF images")
    parser.add_argument("-output", required=True, help="Output folder for converted RGB images")
    parser.add_argument("-workers", type=int, default=8, help="Number of parallel workers (default: 8)")
    args = parser.parse_args()

    input_folder = args.input
    output_folder = args.output
    num_workers = args.workers

    os.makedirs(output_folder, exist_ok=True)

    # Gather lists of source and destination TIFF files
    source_files = [f for f in os.listdir(input_folder) if f.lower().endswith(".tif")]
    dest_files = set(f for f in os.listdir(output_folder) if f.lower().endswith(".tif"))

    print(f"Source files: {len(source_files)}")
    print(f"Destination files: {len(dest_files)}")

    # Only process files that are not yet in the destination
    files_to_process = [f for f in source_files if f not in dest_files]
    print(f"Files to process: {len(files_to_process)}")

    if not files_to_process:
        print("All files already converted. Nothing to do.")
        return

    # Use ThreadPoolExecutor for faster processing
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        for f in files_to_process:
            executor.submit(convert_bgr_to_rgb, f, input_folder, output_folder)

if __name__ == "__main__":
    main()
