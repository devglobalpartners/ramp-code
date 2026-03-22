#################################################################
#
# created for ramp project, April 2024
# Author: Reut Keller, kellerr@who.int
#
#################################################################

import rasterio
from rasterio.mask import mask
from shapely.geometry import mapping, Polygon
import geopandas as gpd
import shutil
import glob as glob
import os, argparse, sys

#TODO add clean code, import constants from module constants
# from filter_rasters import CONSTANTS

def get_raster_intersects_with_vector(rasters_path = str , #path to the directory that has all the downlaoded rasters
                                      polygon_path = str , #path to the polygon that will be used for cropping the raster
                                      dest_raster_path = str , # destination path for results 
                            ):
    lst_paths = glob.glob('/home/jovyan/'+rasters_path+'/*.tif')
    
    polygon = gpd.read_file('/home/jovyan/'+polygon_path)
    

    # Check if destination folder exists, create it if it doesn't
    if not os.path.exists('/home/jovyan/'+dest_raster_path):
        os.makedirs('/home/jovyan/'+dest_raster_path)
    count = 0
    for path in lst_paths:
        
        raster = rasterio.open(path)
        
        # Get the raster's bounding box coordinates
        left, bottom, right, top = raster.bounds
        

        # Get the raster's bounding box coordinates
        left, bottom, right, top = raster.bounds

        # Create a Polygon from the bounding box coordinates
        extent_raster = Polygon([(left, bottom), (right, bottom), (right, top), (left, top)])

        # Convert the Polygon to a GeoDataFrame
        gdf_raster_extent = gpd.GeoDataFrame(geometry=[extent_raster])

        #check if dataframes intersects
        if polygon.geometry.intersects(gdf_raster_extent.unary_union).any():
            file_name = path.split('/')[-1]
            file_dest_path ='/home/jovyan/'+dest_raster_path + '/' + file_name
            
            shutil.copyfile(path, file_dest_path)
            count = count + 1
            
    print(f"{count} files were copied to {dest_raster_path}")
            
            
    return None
        
    

def main():

    parser = argparse.ArgumentParser(description='''
                    copy rasters to new destination if they overlap with given polygon. 
                    ''',formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-rp', '--rasters_path', type=str, required=True, help=r'Path to rastes that will be filtered')
    parser.add_argument('-gp', '--geom_path', type=str, required=True, help=r'Path to geometry. Rasters that do not fall inside this geometry will be filtered out')
    parser.add_argument('-drp', '--dest_raster_path', type=str, required=True, help=r'new path to save only the match rasters')
    args = parser.parse_args()
    
    get_raster_intersects_with_vector(rasters_path =args.rasters_path,
                                      polygon_path = args.geom_path,
                                      dest_raster_path = args.dest_raster_path)
    
    
    
if __name__=="__main__":
    main()
        
    

    