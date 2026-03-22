#################################################################
#
# created for ramp project, August 2025
# Author: Reut Keller Azulay, kellerr@who.int
#
#################################################################

import numpy as np
import rasterio 
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse

def main():
    parser = argparse.ArgumentParser(description='''
    Renames all files in the specified directory to include a .tif suffix, as
    required by the RAMP code
    
    Example: python remove_nir.py -input_dir path/to/directory -output_dir dest/path
    ''', 
    formatter_class=argparse.RawTextHelpFormatter)

    # required
    parser.add_argument('-input_dir', '--input_dir', type=str,
                        required=True, help=r'Path to input images.')
   
    parser.add_argument('-output_dir', '--output_dir', type=str,
                        required=True, help=r'Path to output images.')
   
    args = parser.parse_args()
    input_dir = args.input_dir
    output_dir = args.output_dir
    
    remove_nir_band(input_dir=input_dir,output_dir=output_dir)
    
    
def remove_nir_band(
    input_dir: str , #the direcotry where images are stored
    output_dir: str, #the directory where the results images to be saved
):
    for filename in tqdm(os.listdir(input_dir), desc="Processing rasters"):
        # for filename in os.listdir(input_dir):
        # print(filename)
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)  
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        src = rasterio.open(input_path)

            
        #in the future, change this so user must define the bands (e.g in case image is not rgb, but bgr)
        if src.count >= 4:
            rgb_data = src.read([1, 2, 3])
            
            
            
            # Copy profile and update band count
            profile = src.profile
            profile.update(count=3)

            # Save as new RGB raster
            with rasterio.open(output_path, "w", **profile) as dst:
                dst.write(rgb_data)
                
            print(f"Saved: {output_path}")
        else:
            print(f"the raster has {src.count} bands. Did not proceed")


if __name__=="__main__":
    main()