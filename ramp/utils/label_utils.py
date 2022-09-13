#################################################################
#
# created for ramp project, August 2022
# Author: carolyn.johnston@dev.global
#
#################################################################


import geopandas as gpd
from shapely.geometry import box

def add_overlap_validpoly_columns(gdf):
    '''
    Add both overlap and valid polygon columns to a geodataframe. 
    This function only detects overlaps between valid polygons.
    '''
    # create temporary geodataframe so I can add an index 't_index' column to it
    # this way I have an index column with a known name
    tdf = gdf.copy(deep=True)
    tdf['t_index'] = range(len(tdf))

    # add the valid polygon column to the dataframe
    validpoly_array = [geom.is_valid if geom is not None else 1 for geom in gdf['geometry']]
    tdf['is_valid'] = validpoly_array

    # filter invalid polygons out before checking for overlaps
    valid_tdf = tdf[tdf['is_valid'] == 1]
    
    # do an inner self-join (using 'overlaps' instead of 'intersects').
    joindf= valid_tdf.sjoin(valid_tdf, how="inner", predicate="overlaps")

    # create a list the length of the dataframe that is 1 for every geometry that overlaps another, and 0 otherwise. 
    olap_ids = list(joindf["t_index_left"])
    olap_array = [1 if a in olap_ids else 0 for a in range(len(tdf))]

    # add the new columns to the dataframe
    gdf['is_valid'] = validpoly_array
    gdf["overlaps"] = olap_array
    
    return gdf

def add_overlaps_column(gdf):
    '''
    Add a column to a geodataframe that is 1 if the polygon intersects another polygon, and 0 otherwise.
    '''
    # create temporary geodataframe so I can add an index 't_index' column to it
    # this way I have an index column with a known name

    tdf = gdf.copy(deep=True)
    tdf['t_index'] = range(len(tdf))

    # do an inner self-join on 'overlaps'.
    joindf= tdf.sjoin(tdf, how="inner", predicate="overlaps")

    # create a list the length of the dataframe that is 1 for every geometry that overlaps another, and 0 otherwise. 
    olap_ids = list(joindf["t_index_left"])
    olap_array = [1 if a in olap_ids else 0 for a in range(len(tdf))]

    # add it to the dataframe, return
    gdf["overlaps"] = olap_array
    return gdf


def add_valid_polygon_column(gdf):
    '''
    Add a column to a geodataframe that is 1 if the polygon is valid.
    '''
    validpoly_array = [geom.is_valid for geom in gdf['geometry']]
    gdf['is_valid'] = validpoly_array
    return gdf

def create_tile_gdf (
            raster_tiler, 
            tile_bounds_crs
            ):
    '''
    Create a geojson record of tiles created by the solaris raster and vector tiler.
    '''
    lefts = [tb[0] for tb in raster_tiler.tile_bounds]
    bottoms = [tb[1] for tb in raster_tiler.tile_bounds]
    rights = [tb[2] for tb in raster_tiler.tile_bounds]
    tops = [tb[3] for tb in raster_tiler.tile_bounds]

    tile_records = zip(raster_tiler.tile_paths, lefts, bottoms, rights, tops)

    # construct attributes
    df = gpd.GeoDataFrame(tile_records, columns = ['filename','left', 'bottom', 'right', 'top'])

    # construct geometries
    b = [box(l, b, r, t) for l, b, r, t in zip(df.left, df.bottom, df.right, df.top)]
    gdf = gpd.GeoDataFrame(df, geometry=b)
    gdf = gdf.set_crs(tile_bounds_crs)
    return gdf