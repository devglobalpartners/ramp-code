#################################################################
#
# created for ramp project, August 2022
# Author: carolyn.johnston@dev.global
#
#################################################################

import numpy as np
import geopandas as gpd
from shapely.geometry import Polygon, MultiPolygon
from shapely.geometry.polygon import orient

# adding logging
import logging
log = logging.getLogger(__name__)
log.addHandler(logging.NullHandler())

def fill_rings(the_polygon):
    '''
    Returns the polygon with interiors removed.
    '''
    if isinstance(the_polygon, MultiPolygon):
        return the_polygon
    if the_polygon.interiors:
        return Polygon(list(the_polygon.exterior.coords))
    else:
        return the_polygon

def gpd_add_area_perim(geodf):
    '''
    given a geodataframe containing a column named 'geometry' with polygons, 
    add an area and a perimeter field to it, and return it. 
    :param geopandas.DataFrame geodf: a geodataframe with a polygon field named 'geometry'.
    '''
    if not (geodf.crs and geodf.crs.is_geographic):
        log.error('geodataframe does not have a geographic coordinate system')
        raise TypeError('the geodataframe should have a geographic coordinate system')
        
    geod = geodf.crs.get_geod()

    # Note: I'm defining these functions internally to gpd_add_area_perim, 
    # so they have access to the geod object.

    def area_calc(geom):
        if geom.geom_type not in ['MultiPolygon','Polygon']:
            log.warning(f"geometry type is {geom.geom_type} -- must be one of MultiPolygon, Polygon")
            return np.nan
        
        # For MultiPolygon do each separately
        if geom.geom_type=='MultiPolygon':
            return np.sum([area_calc(p) for p in geom.geoms])

        # orient to ensure a counter-clockwise traversal. 
        # See https://pyproj4.github.io/pyproj/stable/api/geod.html
        # geometry_area_perimeter returns (area, perimeter)
        return geod.geometry_area_perimeter(orient(geom, 1))[0]

    def perim_calc(geom):
        if geom.geom_type not in ['MultiPolygon','Polygon']:
            log.warning(f"geometry type is {geom.geom_type} -- must be one of MultiPolygon, Polygon")
            return np.nan
        
        # For MultiPolygon do each separately
        if geom.geom_type=='MultiPolygon':
            return np.sum([perim_calc(p) for p in geom.geoms])

        # orient to ensure a counter-clockwise traversal. 
        # See https://pyproj4.github.io/pyproj/stable/api/geod.html
        # geometry_area_perimeter returns (area, perimeter)
        return geod.geometry_area_perimeter(orient(geom, 1))[1]

    
    area_series = geodf.geometry.apply(area_calc)
    perim_series = geodf.geometry.apply(perim_calc)
    geodf['area_m2'] = area_series
    geodf['perim_m'] = perim_series
    return geodf


def get_polygon_indices_to_merge(df, pairs_df):
    '''
    used in 'get_labels_from_masks.py' to identify polygons that should be merged 
    from the combined outputs of chipwise polygonization.

    df: geopandas.GeoDataFrame: containing polygons to be merged
    pairs_df: geopandas.GeoDataFrame: containing the result of a spatial join between df and buffered df,
        and postfiltered to retain only boundary-crossing pairs of polygons
    '''

    equivalence_classes = []
    for i1, row1 in pairs_df.iterrows():
        found_class=False
        polyid_left = row1["polyid_left"]
        polyid_right = row1["polyid_right"]

        # if polyid_left == 324 or polyid_right== 324:
        #     print("stop")

        # loop over equivalence classes
        for a_class in equivalence_classes:
            if polyid_left in a_class or polyid_right in a_class:
                a_class.update([polyid_right, polyid_left])
                found_class = True
                break # out of equivalence class loop
 
        # if found_class is false, you've discovered a new equivalence class of polygons to merge
        if not found_class:
            new_class = set([polyid_left, polyid_right])
            equivalence_classes.append(new_class)

    # Create the 'merge_class' attribute 
    # every polygon is in its own equivalence class to start with
    eq_class_list = df["polyid"].copy()

    # start defining new equivalence class indices greater than polyid values
    for ii, eq_class in enumerate(equivalence_classes):
        current_eq_class_index = len(eq_class_list) + ii

        # set all polyids in the same equivalence class to the same merge class
        for polyid in eq_class:
            eq_class_list[polyid] = current_eq_class_index
    
    df["merge_class"]= eq_class_list
    return df
                    
