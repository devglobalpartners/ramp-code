#################################################################
#
# created for ramp project, August 2025
# Author: Reut Keller Azulay, kellerr@who.int
#
#################################################################
import os
import time
import argparse
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import rasterio

def process_file(input_path, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with rasterio.open(input_path) as src:
        if src.count >= 4:
            rgb_data = src.read([1, 2, 3])
            profile = src.profile
            profile.update(count=3)
            with rasterio.open(output_path, "w", **profile) as dst:
                dst.write(rgb_data)
            return f"Saved: {output_path}"
        else:
            return f"Skipped: {input_path}, only {src.count} bands."

# Top-level helper function for ProcessPoolExecutor
def process_file_star(args):
    return process_file(*args)

def remove_nir_band(input_dir: str, output_dir: str, max_workers: int = 4):
    start_time = time.time()
    os.makedirs(output_dir, exist_ok=True)
    files = [f for f in os.listdir(input_dir) if f.lower().endswith(".tif")]

    # Only process files not already in output_dir
    args_list = [(os.path.join(input_dir, f), os.path.join(output_dir, f))
                 for f in files if not os.path.exists(os.path.join(output_dir, f))]

    if not args_list:
        print("All files already processed")
        return

    results = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Use top-level function instead of lambda
        for res in tqdm(executor.map(process_file_star, args_list),
                        total=len(args_list), desc="Processing rasters"):
            results.append(res)

    end_time = time.time()
    for r in results:
        print(r)
    print(f"\nFinished processing {len(args_list)} files with {max_workers} workers in {end_time - start_time:.2f} seconds.")

def main():
    parser = argparse.ArgumentParser(description='''
    Removes NIR band from raster images, keeping only RGB bands.
    
    Example: python remove_nir.py -input_dir path/to/directory -output_dir dest/path
    ''', formatter_class=argparse.RawTextHelpFormatter)

    # Required arguments
    parser.add_argument('-input_dir', '--input_dir', type=str, required=True,
                        help=r'Path to input images.')
    parser.add_argument('-output_dir', '--output_dir', type=str, required=True,
                        help=r'Path to output images.')
    parser.add_argument('-workers', '--workers', type=int, default=4,
                        help='Number of parallel workers.')

    args = parser.parse_args()
    remove_nir_band(input_dir=args.input_dir,
                    output_dir=args.output_dir,
                    max_workers=args.workers)

if __name__=="__main__":
    main()
