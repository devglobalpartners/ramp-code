#################################################################
#
# created for ramp project, April 2024
# Author: Reut Keller , kellerr@who.int
#
#################################################################


import rasterio
from rasterio.mask import mask
from shapely.geometry import mapping, Polygon
import geopandas as gpd
import shutil
import glob as glob
import numpy as np
import pathlib
import matplotlib.pyplot as plt
import os, argparse, sys
from tqdm import tqdm

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
                    
    return print('sucessfuly saved at:{}'.format(savedir))


def crop_rasters_with_polygon(rasters_path : str , # path to rasters to be clipped
                              polygon_path : str , #path to polygon to 
                              dest_raster_path : str , #destination path for 
                             ):
    
    lst_rasters = glob.glob('/home/jovyan/'+rasters_path+'/*.tif')
    gdf_polygon = gpd.read_file("/home/jovyan/"+polygon_path)
    
    for path in tqdm(lst_rasters):
    
        src = rasterio.open(path)
        try:
            masked_raster, transform = mask(src, gdf_polygon.geometry, invert=False)

            file_name = path.split('/')[-1]
            file_dest_path ='/home/jovyan/'+dest_raster_path 

            save_raster(
                img = masked_raster,
                number_of_bands = 3,
                transf = transform,
                paths = file_dest_path,
                tiff_name = path.split('/')[-1],
                crs_img = src.crs)
        except Exception as e:
            print(f"error at: {path} , error: {e}")
            continue

    
    return print('Mission accomplished!')



def main():

    parser = argparse.ArgumentParser(description='''
                    copy rasters to new destination if they overlap with given polygon. 
                    ''',formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-rp', '--rasters_path', type=str, required=True, help=r'Path to rastes that will be filtered')
    parser.add_argument('-gp', '--geom_path', type=str, required=True, help=r'Path to geometry. Rasters that do not fall inside this geometry will be filtered out')
    parser.add_argument('-drp', '--dest_raster_path', type=str, required=True, help=r'new path to save only the match rasters')
    args = parser.parse_args()
    
    crop_rasters_with_polygon(rasters_path =args.rasters_path,
                                      polygon_path = args.geom_path,
                                      dest_raster_path = args.dest_raster_path)
    
    
    
if __name__=="__main__":
    main()

