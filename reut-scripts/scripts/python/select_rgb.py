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



def main():
    parser = argparse.ArgumentParser(description='''
    Save tif as new image with selected bands
    
    Example: python select_rgb.py -input_dir path/to/directory -output_dir dest/path
    ''', 
    formatter_class=argparse.RawTextHelpFormatter)

    # required
    parser.add_argument('-input_dir', '--input_dir', type=str,
                        required=True, help=r'Path to input images.')
   
    parser.add_argument('-output_dir', '--output_dir', type=str,
                        required=True, help=r'Path to output images.')
    
    parser.add_argument('-r_band', '--red', type=int,
                        required=True, help=r'Band number for red')

        
    parser.add_argument('-g_band', '--green', type=int,
                        required=True, help=r'Band number for green')
    
    parser.add_argument('-b_band', '--blue', type=int,
                        required=True, help=r'Band number for blue')
    
    
    args = parser.parse_args()
    input_dir = args.input_dir
    output_dir = args.output_dir
    r_band = args.r_band
    g_band = args.g_band
    b_band = args.b_band
    
    select_bands_rgb(input_dir=input_dir, output_dir=output_dir, r_band=r_band,g_band=g_band,b_band=b_band)
    
    

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
                                   driver='COG',
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


def select_bands_rgb(
    input_dir: str , #the direcotry where images are stored
    output_dir: str, #the directory where the results images to be saved
    r_band: int ,
    g_band: int,
    b_band: int,
    
):
    for filename in tqdm(os.listdir(input_dir), desc="Processing rasters"):
        # for filename in os.listdir(input_dir):
        # print(filename)
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)  
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)



        with rasterio.open(input_path) as src:
            arr = src.read()
            
            # Convert to zero-based indices
            bands = [r_band - 1, g_band - 1, b_band - 1]
            
            # Validate bands
            for b in bands:
                if b < 0 or b >= src.count:
                    raise ValueError(f"Band index {b+1} is out of range (1-{src.count}) in {filename}")
            
            # Extract selected bands
            selected_data = arr[bands, :, :]
            
       
            
            save_raster(
                img = selected_data,
                number_of_bands = 3, 
                transf = src.transform,
                paths = output_dir,
                tiff_name = filename,
                crs_img= src.crs)


if __name__=="__main__":
    main()