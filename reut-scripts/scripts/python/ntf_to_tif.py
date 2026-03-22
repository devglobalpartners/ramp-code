#################################################################
#
# created for ramp project, September 2025
# Author: Reut Keller Azulay, kellerr@who.int
#
#################################################################

import pathlib
import rasterio
import os
from tqdm import tqdm
import numpy as np

from tqdm import tqdm
import argparse

def main():
    parser = argparse.ArgumentParser(description='''
    Renames all files in the specified directory to include a .tif suffix, as
    required by the RAMP code
    
    Example: python ntf_to_tif.py -input_dir path/to/directory -output_dir dest/path
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
    
    ntf_to_tif(input_dir=input_dir,output_dir=output_dir)
    

def save_raster(
    img:np.ndarray , # ndarray image
    number_of_bands: int , #number of  bands
    transf:tuple ,#transform function of the raster
    paths:str ,#save path raster
    tiff_name:str ,#name of specific file
    crs_img:rasterio.crs.CRS,
    ):  
    
    '''
    Function to save raster in requested path

    '''

    savedir=pathlib.Path(paths)
    savedir.mkdir(parents=True,exist_ok=True)


    with rasterio.open(savedir /tiff_name, 
                                   'w',
                                   driver='GTiff',
                                   height=img.shape[1],
                                   width=img.shape[2],
                                   count=number_of_bands,
                                   dtype=img.dtype,
                                   crs=crs_img,
                                   nodata=None, # change if data has nodata value
                                   transform=transf) as dst:

            for n in np.arange(0,number_of_bands):
                  dst.write(img[n,:,:],int(n+1))   
                    
    return None
def ntf_to_tif(
    input_dir: str , #the direcotry where images are stored
    output_dir: str, #the directory where the results images to be saved
):
    for filename in tqdm(os.listdir(input_dir), desc="Processing rasters"):
    # for filename in os.listdir(input_dir):
        print(filename)
        # process only files with .NTF extension
            
        output_filename = os.path.splitext(filename)[0] + ".tif"
        input_path = os.path.join(input_dir, filename)

        if not os.path.isfile(input_path) or not filename.lower().endswith(".ntf"):
            print(f"file {filename} is not .NTF, skipping")
            continue
       
        output_path = os.path.join(output_dir, output_filename)  
            
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        print(f"reading {filename}")
        src = rasterio.open(input_path)
        
        arr=src.read()
        print(arr.shape)
        
        save_raster(
            img= arr,
            number_of_bands=3,
            transf=src.transform,
            paths=output_dir,
            tiff_name=output_filename,
            crs_img=src.crs)
        print(f"saved to: {output_path}")
        
if __name__=="__main__":
    main()