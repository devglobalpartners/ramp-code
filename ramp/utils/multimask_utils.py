
#################################################################
#
# created for ramp project, August 2022
#
#################################################################


### Forked from solaris sol.vector.mask on 20220701

# adding logging
import logging
log = logging.getLogger(__name__)
log.addHandler(logging.NullHandler())

from solaris.utils.core import _check_df_load, _check_gdf_load, _check_geom, _check_crs
from solaris.utils.core import _check_skimage_im_load, _check_rasterio_im_load
from solaris.utils.geo import gdf_get_projection_unit, reproject
from solaris.utils.geo import geometries_internal_intersection
from solaris.utils.tile import save_empty_geojson

# note that affine_transform_gdf from this import will be overwritten by the local version.
# replace with import by name!
from solaris.vector.polygon import convert_poly_coords, geojson_to_px_gdf
import numpy as np
from shapely.geometry import shape
from shapely.geometry import Polygon
import geopandas as gpd
import pandas as pd
import rasterio
from rasterio import features
from affine import Affine
from skimage.morphology import square, erosion, dilation
import os
from tqdm.auto import tqdm

def df_to_px_mask(
                df, 
                channels=['footprint'], 
                out_file=None, 
                reference_im=None,
                geom_col='geometry', 
                do_transform=None, 
                affine_obj=None,
                shape=(900, 900), 
                out_type='int', 
                burn_value=255, 
                **kwargs
                ):
    """Convert a dataframe of geometries to a pixel mask.

    Arguments
    ---------
    df : :class:`pandas.DataFrame` or :class:`geopandas.GeoDataFrame`
        A :class:`pandas.DataFrame` or :class:`geopandas.GeoDataFrame` instance
        with a column containing geometries (identified by `geom_col`). If the
        geometries in `df` are not in pixel coordinates, then `affine` or
        `reference_im` must be passed to provide the transformation to convert.
    channels : list, optional
        The mask channels to generate. There are three values that this can
        contain:

        - ``"footprint"``: Create a full footprint mask, with 0s at pixels
            that don't fall within geometries and `burn_value` at pixels that
            do.
        - ``"boundary"``: Create a mask with geometries outlined. Use
            `boundary_width` to set how thick the boundary will be drawn.
        - ``"contact"``: Create a mask with regions between >= 2 closely
            juxtaposed geometries labeled. Use `contact_spacing` to set the
            maximum spacing between polygons to be labeled.

        Each channel correspond to its own `shape` plane in the output.
    out_file : str, optional
        Path to an image file to save the output to. Must be compatible with
        :class:`rasterio.DatasetReader`. If provided, a `reference_im` must be
        provided (for metadata purposes).
    reference_im : :class:`rasterio.DatasetReader` or `str`, optional
        An image to extract necessary coordinate information from: the
        affine transformation matrix, the image extent, etc. If provided,
        `affine_obj` and `shape` are ignored.
    geom_col : str, optional
        The column containing geometries in `df`. Defaults to ``"geometry"``.
    do_transform : bool, optional
        Should the values in `df` be transformed from geospatial coordinates
        to pixel coordinates? Defaults to ``None``, in which case the function
        attempts to infer whether or not a transformation is required based on
        the presence or absence of a CRS in `df`. If ``True``, either
        `reference_im` or `affine_obj` must be provided as a source for the
        the required affine transformation matrix.
    affine_obj : `list` or :class:`affine.Affine`, optional
        Affine transformation to use to convert from geo coordinates to pixel
        space. Only provide this argument if `df` is a
        :class:`geopandas.GeoDataFrame` with coordinates in a georeferenced
        coordinate space. Ignored if `reference_im` is provided.
    shape : tuple, optional
        An ``(x_size, y_size)`` tuple defining the pixel extent of the output
        mask. Ignored if `reference_im` is provided.
    burn_value : `int` or `float`
        The value to use for labeling objects in the mask. Defaults to 255 (the
        max value for ``uint8`` arrays). The mask array will be set to the same
        dtype as `burn_value`.
    kwargs
        Additional arguments to pass to `boundary_mask` or `contact_mask`. See
        those functions for requirements.

    Returns
    -------
    mask : :class:`numpy.array`
        A pixel mask with 0s for non-object pixels and `burn_value` at object
        pixels. `mask` dtype will coincide with `burn_value`. Shape will be
        ``(shape[0], shape[1], len(channels))``, with channels ordered per the
        provided `channels` `list`.

    """
    if isinstance(channels, str):  # e.g. if "contact", not ["contact"]
        channels = [channels]

    if out_file and not reference_im:
        raise ValueError(
            'If saving output to file, `reference_im` must be provided.')

    mask_dict = {}

    meters = kwargs.get('meters',False)

    # 20220802 bugfix: clean up geometries before buffer_df_geoms is called
    df = buffer_df_geoms(df, 0, meters=meters, reference_im=reference_im)


    if 'footprint' in channels:
        mask_dict['footprint'] = shapes_mask(
            df=df, 
            reference_im=reference_im, 
            geom_col=geom_col,
            do_transform=do_transform, 
            shape=shape,
            burn_value=burn_value,
            out_file=None
        )
    if 'boundary' in channels:

        try:
            interior_df = get_interior_dataframe(
                            df, 
                            geom_col=geom_col,
                            reference_im=reference_im,
                            meters=kwargs.get('meters',False),
                            boundary_width=kwargs.get('boundary_width', 3),
                            out_file=None
            )
                            # out_file="/tf/ramp-data/temp.file.geojson")

            interior_mask = shapes_mask(
                            df=interior_df, 
                            reference_im=reference_im, 
                            geom_col=geom_col,
                            do_transform=do_transform, 
                            shape=shape,
                            burn_value=burn_value,
                            out_file=None
                            )
                            # out_file="/tf/ramp-data/temp.file.tif")

            mask_dict['boundary'] = mask_dict['footprint'] - interior_mask

        # handle these key errors that arise when there is a problem with negative buffering
        except KeyError:
            log.info(f"could not determine boundary pixels in df_to_px_mask (got KeyError), bypassing")
            mask_dict["boundary"] = np.zeros(shape)

    if 'contact' in channels:

        try:
            contact_df = get_contact_dataframe(
                            df,
                            geom_col=geom_col,
                            reference_im=reference_im,
                            meters=kwargs.get('meters', False),
                            contact_spacing=kwargs.get('contact_spacing', 8)
                            )


            mask_dict['contact'] = shapes_mask(
                            df=contact_df, 
                            reference_im=reference_im, 
                            geom_col=geom_col,
                            do_transform=do_transform, 
                            shape=shape,
                            burn_value=burn_value,
                            out_file=None)

        except KeyError:
            log.info(f"could not determine close contact pixels in df_to_px_mask (got KeyError), bypassing")
            mask_dict["contact"] = np.zeros(shape)
            
    output_arr = np.stack([mask_dict[c] for c in channels], axis=-1)

    if reference_im:
        reference_im = _check_rasterio_im_load(reference_im)
    if out_file:
        meta = reference_im.meta.copy()
        meta.update(count=output_arr.shape[-1])
        meta.update(dtype='uint8')
        with rasterio.open(out_file, 'w', **meta) as dst:
            # I hate band indexing.
            for c in range(1, 1 + output_arr.shape[-1]):
                dst.write(output_arr[:, :, c-1], indexes=c)

    return output_arr

    ##############################################################################

def shapes_mask(df, 
                out_file, 
                reference_im, 
                geom_col,
                do_transform,
                shape,
                burn_value):
    """Convert a dataframe of geometries to a pixel mask.

    Arguments
    ---------
    df : :class:`pandas.DataFrame` or :class:`geopandas.GeoDataFrame`
        A :class:`pandas.DataFrame` or :class:`geopandas.GeoDataFrame` instance
        with a column containing geometries (identified by `geom_col`). If the
        geometries in `df` are not in pixel coordinates, then `affine` or
        `reference_im` must be passed to provide the transformation to convert.
    out_file : str, optional
        Path to an image file to save the output to. Must be compatible with
        :class:`rasterio.DatasetReader`. If provided, a `reference_im` must be
        provided (for metadata purposes).
    reference_im : :class:`rasterio.DatasetReader` or `str`, optional
        An image to extract necessary coordinate information from: the
        affine transformation matrix, the image extent, etc. 
    geom_col : str, optional
        The column containing geometries in `df`. Defaults to ``"geometry"``.
    do_transform : bool, optional
        Should the values in `df` be transformed from geospatial coordinates
        to pixel coordinates? 
        Defaults to ``None``, in which case the function
        attempts to infer whether or not a transformation is required based on
        the presence or absence of a CRS in `df`. 
        If ``True``, either
        `reference_im` or `affine_obj` must be provided as a source for the
        the required affine transformation matrix.
    burn_value : `int` or `float`, optional
        The value to use for labeling objects in the mask. Defaults to 255 (the
        max value for ``uint8`` arrays). The mask array will be set to the same
        dtype as `burn_value`. Ignored if `burn_field` is provided.

    Returns
    -------
    mask : :class:`numpy.array`
        A pixel mask with 0s for non-object pixels and `burn_value` at object
        pixels. `mask` dtype will coincide with `burn_value`.

    """
    # start with required checks and pre-population of values

    if len(df) == 0 and not out_file:
        return np.zeros(shape=shape, dtype='uint8')

    affine_obj = None
    if not do_transform:
        affine_obj = Affine(1, 0, 0, 0, 1, 0)  # identity transform

    shape = None
    if reference_im:
        reference_im = _check_rasterio_im_load(reference_im)
        shape = reference_im.shape
        if do_transform:
            affine_obj = reference_im.transform

    # extract geometries 
    feature_list = list(df[geom_col])

    if len(df) > 0:
        output_arr = features.rasterize(shapes=feature_list, 
                                        fill=0,
                                        default_value=burn_value,
                                        out_shape=shape,
                                        transform=affine_obj,
                                        dtype="uint8")
    else:
        output_arr = np.zeros(shape=shape, dtype='uint8')
    
    if out_file:
        meta = reference_im.meta.copy()
        meta.update(count=1)
        meta.update(dtype='uint8')
        meta.update(nodata=None)
        with rasterio.open(out_file, 'w', **meta) as dst:
            output_mask = np.expand_dims(output_arr, axis=0)
            dst.write(output_mask)

    return output_arr

##################################################################################################
## CJ 20220705: convert footprint-bdry-contact mask to sparse mask

def multimask_to_sparse_multimask(onehot_mask):
    """
    Change a one-hot (channels-last) multimask to a sparse-encoded multimask
    bdry > footprint > contact
    onehot_mask: numpy ndarray of shape [H,W, C] with values 0,1
    sparse_mask: numpy ndarray of shape [H,W,1] with values 0,...,C-1 
    """
    sparse_mask = np.zeros(onehot_mask.shape[:2], dtype=np.uint8)
    sparse_mask[np.where(onehot_mask[:,:,2] > 0)] = 3 #contact
    sparse_mask[np.where(onehot_mask[:,:,0] > 0)] = 1 #footprint
    sparse_mask[np.where(onehot_mask[:,:,1] > 0)] = 2 #boundary
    return np.expand_dims(sparse_mask, axis=-1)


## CJ 20220704: get shapes to burn into rasters for boundary and contact points. 
def get_interior_dataframe(
                df, 
                boundary_width, 
                reference_im, 
                meters,
                geom_col="geometry",
                out_file=None
                ):
    '''
    get a dataframe of negative-buffered building masks. 
    The difference between the original mask, and this mask, becomes the building boundary mask. -
    '''
    interior_df = buffer_df_geoms(
                        df, 
                        buffer=-boundary_width, 
                        reference_im=reference_im, 
                        meters=meters
                        )

    # remove empty geometries
    interior_df = interior_df[~interior_df.is_empty]

    # return interior df
    if out_file:
        interior_df.to_file(out_file)
    return interior_df


def get_contact_dataframe(
                    df,
                    contact_spacing, 
                    reference_im,
                    meters, 
                    geom_col
                    ):
    '''
    '''
    df['t_index'] = range(len(df))

    # create a new copy of the df with buffered geometries.
    buffer_df = buffer_df_geoms(
                    df, 
                    buffer=contact_spacing,
                    reference_im=reference_im
                    )

    # do a spatial inner join of buffered geometries with unbuffered geometries.
    joined_df = buffer_df.overlay(df, how="intersection")

    # remove the self-intersections
    joined_df = joined_df[~(joined_df['t_index_1']==joined_df["t_index_2"])]
    
    # remove empty polygons
    joined_df = joined_df[~joined_df.is_empty]

    # If B(X) represents the buffered version of shape X, then
    # this df contains shapes of the form B(X)\intersect Y for all X, Y in df
    # if B(X)\intersect Y is in the df, so is B(Y)\intersect X
    joined_df = joined_df[['t_index_1', 't_index_2', "geometry"]]

    # eject if there are no nontrivial polygons
    if len(joined_df) == 0:
        return joined_df

        
    # next: if B(X)\intersect Y is in the df, then we want to get B(X)\intersect B(Y)
    # we need to index into the data frame of buffered shapes, which is tricky to get right.
    # This function will be applied to rows in joined_df using 'apply'.
    # It returns the polygon in buffer_df pointed to by t_index_1.
    def get_buff_geom_1(x): 
        return buffer_df[buffer_df["t_index"]==x.t_index_1]["geometry"].values[0]

    # This one returns the polygon in buffer_df pointed to by t_index_2.
    def get_buff_geom_2(x): 
        return buffer_df[buffer_df["t_index"]==x.t_index_2]["geometry"].values[0]

    # Now we apply the lookups to get the buffered polygons.
    b1 = joined_df.apply(lambda x: get_buff_geom_1(x), axis=1  )
    b2 = joined_df.apply(lambda x: get_buff_geom_2(x), axis=1)

    # Note: b1 and b2 are pandas series which happen to contain polygons! 
    # they are not GeoSeries so you can't run geo operations on them.
    # here I turn them into GeoSeries so I can call 'intersection' on them
    buff1 = gpd.GeoSeries(b1, crs=joined_df.crs )
    buff2 = gpd.GeoSeries(b2, crs=joined_df.crs)

    # this returns the intersection polygons, which cover the sets of contact points. 
    # Note that since B(X)\intersect B(Y)=B(Y)\intersect B(X),
    # each polygon will appear twice, but we don't mind. 
    buffer_ints = buff1.intersection(buff2, align=True)

    # Construct a geodataframe with the intersection polygons 
    # and the indices in the original dataset. Return it. 
    gdf_data = {
        "t1":joined_df["t_index_1"],
        "t2":joined_df["t_index_2"],
        "geometry":buffer_ints
    }
    contactpts_df = gpd.GeoDataFrame(data=gdf_data, crs=joined_df.crs)
    return contactpts_df

        
##########################################################################################
## ORIGINAL SOLARIS
##########################################################################################


def affine_transform_gdf(gdf, affine_obj, inverse=False, geom_col="geometry",
                         precision=None):
    """Perform an affine transformation on a GeoDataFrame.
    
    Originally from solaris repository but contains a bugfix.

    Arguments
    ---------
    gdf : :class:`geopandas.GeoDataFrame`, :class:`pandas.DataFrame`, or `str`
        A GeoDataFrame, pandas DataFrame with a ``"geometry"`` column (or a
        different column containing geometries, identified by `geom_col` -
        note that this column will be renamed ``"geometry"`` for ease of use
        with geopandas), or the path to a saved file in .geojson or .csv
        format.
    affine_obj : list or :class:`affine.Affine`
        An affine transformation to apply to `geom` in the form of an
        ``[a, b, d, e, xoff, yoff]`` list or an :class:`affine.Affine` object.
    inverse : bool, optional
        Use this argument to perform the inverse transformation.
    geom_col : str, optional
        The column in `gdf` corresponding to the geometry. Defaults to
        ``'geometry'``.
    precision : int, optional
        Decimal precision to round the geometries to. If not provided, no
        rounding is performed.
    """
    if isinstance(gdf, str):  # assume it's a geojson
        if gdf.lower().endswith('json'):
            gdf = gpd.read_file(gdf)
        elif gdf.lower().endswith('csv'):
            gdf = pd.read_csv(gdf)
        else:
            raise ValueError(
                "The file format is incompatible with this function.")
    if 'geometry' not in gdf.columns:
        gdf = gdf.rename(columns={geom_col: 'geometry'})


    # Check datatype of gdf["geometry"]: if it's not a Polygon, 
    # assume its a WKT string and load it.
    
    # CJ bugfix 20220728: indexing into gdf["geometry"] is done with the GeoSeries index,
    # not with the list index! So if list index 0 is not present, a call to 
    # gdf["geometry"][0] will throw a KeyError. 
    # I get around it by retrieving its keys and checking the type of the first one. 
#    if not isinstance(gdf['geometry'][0], Polygon):

    gdf_keys = gdf['geometry'].keys()
    first_key = gdf_keys[0]
    if not isinstance(gdf['geometry'][first_key], Polygon):
        gdf['geometry'] = gdf['geometry'].apply(shapely.wkt.loads)

    gdf["geometry"] = gdf["geometry"].apply(convert_poly_coords,
                                            affine_obj=affine_obj,
                                            inverse=inverse)
    if precision is not None:
        gdf['geometry'] = gdf['geometry'].apply(
            _reduce_geom_precision, precision=precision)

    # the CRS is no longer valid - remove it
    gdf.crs = None

    return gdf


def buffer_df_geoms(df, buffer, meters=False, reference_im=None,
                    geom_col='geometry', affine_obj=None):
    """Buffer geometries within a pd.DataFrame or gpd.GeoDataFrame.

    Arguments
    ---------
    df : :class:`pandas.DataFrame` or :class:`geopandas.GeoDataFrame`
        A :class:`pandas.DataFrame` or :class:`geopandas.GeoDataFrame` instance
        with a column containing geometries (identified by `geom_col`). If `df`
        lacks a ``crs`` attribute (isn't a :class:`geopandas.GeoDataFrame` )
        and ``meters=True``, then `reference_im` must be provided for
        georeferencing.
    buffer : `int` or `float`
        The amount to buffer the geometries in `df`. In pixel units unless
        ``meters=True``. This corresponds to width/2 in mask creation
        functions.
    meters : bool, optional
        Should buffers be in pixel units (default) or metric units (if `meters`
        is ``True``)?
    reference_im : `str` or :class:`rasterio.DatasetReader`, optional
        The path to a reference image covering the same geographic extent as
        the area labeled in `df`. Provided for georeferencing of pixel
        coordinate geometries in `df` or conversion of georeferenced geometries
        to pixel coordinates as needed. Required if `meters` is ``True`` and
        `df` lacks a ``crs`` attribute.
    geom_col : str, optional
        The column containing geometries in `df`. Defaults to ``"geometry"``.
    affine_obj : `list` or :class:`affine.Affine`, optional
        Affine transformation to use to convert geoms in `df` from a geographic
        crs to pixel space. Only provide this argument if `df` is a
        :class:`geopandas.GeoDataFrame` with coordinates in a georeferenced
        coordinate space. Ignored if `reference_im` is provided.

    Returns
    -------
    buffered_df : :class:`pandas.DataFrame`
        A :class:`pandas.DataFrame` in the original coordinate reference system
        with objects buffered per `buffer`.

    See Also
    --------
    road_mask : Function to create road network masks.
    contact_mask : Function to create masks of contact points between polygons.
    """

    df = df[~df.is_empty]

    if len(df)==0:
        return df.copy()

    if reference_im is not None:
        reference_im = _check_rasterio_im_load(reference_im)

    if hasattr(df, 'crs'):
        orig_crs = _check_crs(df.crs)
    else:
        orig_crs = None  # will represent pixel crs

    # Check if dataframe is in the appropriate units and reproject if not
    if not meters:
        if hasattr(df, 'crs') and reference_im is not None:
            # if the df is georeferenced and a reference_im is provided,
            # use reference_im to transform df to px coordinates
            df_for_buffer = geojson_to_px_gdf(df.copy(), reference_im)
        elif hasattr(df, 'crs') and reference_im is None:

            # CJ 20220728: bugfix: inverse needs to be used here
            df_for_buffer = affine_transform_gdf(df.copy(),
                                                 affine_obj=affine_obj,
                                                 inverse=True)
        else:  # if it's already in px coordinates
            df_for_buffer = df.copy()

    else:
        # check if the df is in a metric crs
        if hasattr(df, 'crs'):
            if crs_is_metric(df):
                df_for_buffer = df.copy()
            else:
                df_for_buffer = reproject(df.copy())  # defaults to UTM
        else:
            # assume df is in px coords - use reference_im to georegister
            if reference_im is not None:
                df_for_buffer = georegister_px_df(df.copy(),
                                                  im_path=reference_im)
            else:
                raise ValueError('If using `meters=True`, either `df` must be '
                                 'a geopandas GeoDataFrame or `reference_im` '
                                 'must be provided for georegistration.')

    df_for_buffer[geom_col] = df_for_buffer[geom_col].apply(
        lambda x: x.buffer(buffer))

    # sometimes negative buffering can produce multipolygons
    df_for_buffer = df_for_buffer.explode(ignore_index=True) 

    # return to original crs
    if _check_crs(getattr(df_for_buffer, 'crs', None)) != orig_crs:
        if orig_crs is not None and \
                getattr(df_for_buffer, 'crs', None) is not None:
            buffered_df = df_for_buffer.to_crs(orig_crs.to_wkt())
        elif orig_crs is None:  # but df_for_buffer has one: meters=True case
            buffered_df = geojson_to_px_gdf(df_for_buffer, reference_im)
        else:  # orig_crs exists, but df_for_buffer doesn't have one
            buffered_df = georegister_px_df(df_for_buffer,
                                            im_path=reference_im,
                                            affine_obj=affine_obj,
                                            crs=orig_crs)
    else:
        buffered_df = df_for_buffer

    return buffered_df


def preds_to_binary(pred_arr, channel_scaling=None, bg_threshold=0):
    """Convert a set of predictions from a neural net to a binary mask.

    Arguments
    ---------
    pred_arr : :class:`numpy.ndarray`
        A set of predictions generated by a neural net (generally in ``float``
        dtype). This can be a 2D array or a 3D array, in which case it will
        be convered to a 2D mask output with optional channel scaling (see
        the `channel_scaling` argument). If a filename is provided instead of
        an array, the image will be loaded using scikit-image.
    channel_scaling : `list`-like of `float`s, optional
        If `pred_arr` is a 3D array, this argument defines how each channel
        will be combined to generate a binary output. channel_scaling should
        be a `list`-like of length equal to the number of channels in
        `pred_arr`. The following operation will be performed to convert the
        multi-channel prediction to a 2D output ::

            sum(pred_arr[channel]*channel_scaling[channel])

        If not provided, no scaling will be performend and channels will be
        summed.

    bg_threshold : `int` or `float`, optional
        The cutoff to set to distinguish between background and foreground
        pixels in the final binary mask. Binarization takes place *after*
        channel scaling and summation (if applicable). Defaults to 0.

    Returns
    -------
    mask_arr : :class:`numpy.ndarray`
        A 2D boolean ``numpy`` array with ``True`` for foreground pixels and
        ``False`` for background.
    """
    pred_arr = _check_skimage_im_load(pred_arr).copy()

    if len(pred_arr.shape) == 3:
        if pred_arr.shape[0] < pred_arr.shape[-1]:
            pred_arr = np.moveaxis(pred_arr, 0, -1)
        if channel_scaling is None:  # if scale values weren't provided
            channel_scaling = np.ones(shape=(pred_arr.shape[-1]),
                                      dtype='float')
        pred_arr = np.sum(pred_arr*np.array(channel_scaling), axis=-1)

    mask_arr = (pred_arr > bg_threshold).astype('uint8')

    return mask_arr*255

def binary_mask_to_geodataframe(binary_array,
                                affine_object=None,
                                df_crs="EPSG:4326", 
                                do_transform=False,
                                simplify=False,
                                tolerance=0.5,
                                min_area=0,
                                **kwargs):
    """
    Get a polygon geodataframe from a binary mask.
    Simplified version of 'mask_to_poly_geojson' that takes an affine object for transformation
    instead of a reference image.

    Arguments
    ---------
    binary_array : :class:`numpy.ndarray`
        A 2D array of integers with values in 0,1.
    affine_object : object of class Affine 
        for calculating the final geotransform.
        Required if 'do_transform' is True
    df_crs: EPSG string for the CRS of the output geodataframe
        Required if 'do_transform' is true
    min_area : int, optional
        The minimum area of a polygon to retain. Filtering is done AFTER
        any coordinate transformation, and therefore will be in destination
        units.
    simplify : bool, optional
        If ``True``, will use the Douglas-Peucker algorithm to simplify edges,
        saving memory and processing time later. Defaults to ``False``.
    tolerance : float, optional
        The tolerance value to use for simplification with the Douglas-Peucker
        algorithm. Defaults to ``0.5``. Only has an effect if
        ``simplify=True``.

    Returns
    -------
    gdf : :class:`geopandas.GeoDataFrame`
        A GeoDataFrame of polygons.

    """

    if do_transform and affine_object is None:
        raise ValueError(
            'Coordinate transformation requires an input affine transformation.')

    if do_transform:
        transform = affine_object
        crs = _check_crs(df_crs)
    else:
        transform = Affine(1, 0, 0, 0, 1, 0)  # identity transform
        crs = rasterio.crs.CRS()

    # pixels not in 'mask' are excluded from feature generation,
    # so we generate only buildings
    mask = binary_array > 0
    mask = mask.astype('uint8')

    polygon_generator = features.shapes(binary_array,
                                        transform=transform,
                                        mask=mask)
    polygons = []
    values = []  # pixel values for the polygon in mask_arr
    for polygon, value in polygon_generator:
        p = shape(polygon).buffer(0.0)
        if p.area >= min_area:
            polygons.append(shape(polygon).buffer(0.0))
            values.append(value)

    polygon_gdf = gpd.GeoDataFrame({'geometry': polygons, 'value': values},
                                   crs=crs.to_wkt())
    if simplify:
        polygon_gdf['geometry'] = polygon_gdf['geometry'].apply(
            lambda x: x.simplify(tolerance=tolerance)
        )

    return polygon_gdf




def mask_to_poly_geojson(pred_arr, channel_scaling=None, reference_im=None,
                         output_path=None, output_type='geojson', min_area=40,
                         bg_threshold=0, do_transform=None, simplify=False,
                         tolerance=0.5, **kwargs):
    """Get polygons from an image mask.

    Arguments
    ---------
    pred_arr : :class:`numpy.ndarray`
        A 2D array of integers. Multi-channel masks are not supported, and must
        be simplified before passing to this function. Can also pass an image
        file path here.
    channel_scaling : :class:`list`-like, optional
        If `pred_arr` is a 3D array, this argument defines how each channel
        will be combined to generate a binary output. channel_scaling should
        be a `list`-like of length equal to the number of channels in
        `pred_arr`. The following operation will be performed to convert the
        multi-channel prediction to a 2D output ::

            sum(pred_arr[channel]*channel_scaling[channel])

        If not provided, no scaling will be performend and channels will be
        summed.
    reference_im : str, optional
        The path to a reference geotiff to use for georeferencing the polygons
        in the mask. Required if saving to a GeoJSON (see the ``output_type``
        argument), otherwise only required if ``do_transform=True``.
    output_path : str, optional
        Path to save the output file to. If not provided, no file is saved.
    output_type : ``'csv'`` or ``'geojson'``, optional
        If ``output_path`` is provided, this argument defines what type of file
        will be generated - a CSV (``output_type='csv'``) or a geojson
        (``output_type='geojson'``).
    min_area : int, optional
        The minimum area of a polygon to retain. Filtering is done AFTER
        any coordinate transformation, and therefore will be in destination
        units.
    bg_threshold : int, optional
        The cutoff in ``mask_arr`` that denotes background (non-object).
        Defaults to ``0``.
    simplify : bool, optional
        If ``True``, will use the Douglas-Peucker algorithm to simplify edges,
        saving memory and processing time later. Defaults to ``False``.
    tolerance : float, optional
        The tolerance value to use for simplification with the Douglas-Peucker
        algorithm. Defaults to ``0.5``. Only has an effect if
        ``simplify=True``.

    Returns
    -------
    gdf : :class:`geopandas.GeoDataFrame`
        A GeoDataFrame of polygons.

    """

    mask_arr = preds_to_binary(pred_arr, channel_scaling, bg_threshold)

    if do_transform and reference_im is None:
        raise ValueError(
            'Coordinate transformation requires a reference image.')

    if do_transform:
        with rasterio.open(reference_im) as ref:
            transform = ref.transform
            crs = ref.crs
            ref.close()
    else:
        transform = Affine(1, 0, 0, 0, 1, 0)  # identity transform
        crs = rasterio.crs.CRS()

    mask = mask_arr > bg_threshold
    mask = mask.astype('uint8')

    polygon_generator = features.shapes(mask_arr,
                                        transform=transform,
                                        mask=mask)
    polygons = []
    values = []  # pixel values for the polygon in mask_arr
    for polygon, value in polygon_generator:
        p = shape(polygon).buffer(0.0)
        if p.area >= min_area:
            polygons.append(shape(polygon).buffer(0.0))
            values.append(value)

    polygon_gdf = gpd.GeoDataFrame({'geometry': polygons, 'value': values},
                                   crs=crs.to_wkt())
    if simplify:
        polygon_gdf['geometry'] = polygon_gdf['geometry'].apply(
            lambda x: x.simplify(tolerance=tolerance)
        )
    # save output files
    if output_path is not None:
        if output_type.lower() == 'geojson':
            if len(polygon_gdf) > 0:
                polygon_gdf.to_file(output_path, driver='GeoJSON')
            else:
                save_empty_geojson(output_path, polygon_gdf.crs.to_epsg())
        elif output_type.lower() == 'csv':
            polygon_gdf.to_csv(output_path, index=False)

    return polygon_gdf


def crs_is_metric(gdf):
    """Check if a GeoDataFrame's CRS is in metric units."""
    units = str(gdf_get_projection_unit(gdf)).strip().lower()
    if units in ['"meter"', '"metre"', "'meter'", "'meter'",
                 'meter', 'metre']:
        return True
    else:
        return False


def _check_do_transform(df, reference_im, affine_obj):
    """Check whether or not a transformation should be performed."""
    try:
        crs = getattr(df, 'crs')
    except AttributeError:
        return False  # if it doesn't have a CRS attribute

    if not crs:
        return False  # return False for do_transform if crs is falsey
    elif crs and (reference_im is not None or affine_obj is not None):
        # if the input has a CRS and another obj was provided for xforming
        return True


def instance_mask(df, out_file=None, reference_im=None, geom_col='geometry',
                  do_transform=None, affine_obj=None, shape=(900, 900),
                  out_type='int', burn_value=255, burn_field=None, nodata_value=0):
    """Convert a dataframe of geometries to a pixel mask.

    Arguments
    ---------
    df : :class:`pandas.DataFrame` or :class:`geopandas.GeoDataFrame`
        A :class:`pandas.DataFrame` or :class:`geopandas.GeoDataFrame` instance
        with a column containing geometries (identified by `geom_col`). If the
        geometries in `df` are not in pixel coordinates, then `affine` or
        `reference_im` must be passed to provide the transformation to convert.
    out_file : str, optional
        Path to an image file to save the output to. Must be compatible with
        :class:`rasterio.DatasetReader`. If provided, a `reference_im` must be
        provided (for metadata purposes).
    reference_im : :class:`rasterio.DatasetReader` or `str`, optional
        An image to extract necessary coordinate information from: the
        affine transformation matrix, the image extent, etc. If provided,
        `affine_obj` and `shape` are ignored.
    geom_col : str, optional
        The column containing geometries in `df`. Defaults to ``"geometry"``.
    do_transform : bool, optional
        Should the values in `df` be transformed from geospatial coordinates
        to pixel coordinates? Defaults to ``None``, in which case the function
        attempts to infer whether or not a transformation is required based on
        the presence or absence of a CRS in `df`. If ``True``, either
        `reference_im` or `affine_obj` must be provided as a source for the
        the required affine transformation matrix.
    affine_obj : `list` or :class:`affine.Affine`, optional
        Affine transformation to use to convert from geo coordinates to pixel
        space. Only provide this argument if `df` is a
        :class:`geopandas.GeoDataFrame` with coordinates in a georeferenced
        coordinate space. Ignored if `reference_im` is provided.
    shape : tuple, optional
        An ``(x_size, y_size)`` tuple defining the pixel extent of the output
        mask. Ignored if `reference_im` is provided.
    out_type : 'float' or 'int'
    burn_value : `int` or `float`, optional
        The value to use for labeling objects in the mask. Defaults to 255 (the
        max value for ``uint8`` arrays). The mask array will be set to the same
        dtype as `burn_value`. Ignored if `burn_field` is provided.
    burn_field : str, optional
        Name of a column in `df` that provides values for `burn_value` for each
        independent object. If provided, `burn_value` is ignored.
    nodata_value : `int` or `float`, optional
        The value to use for nodata pixels in the mask. Defaults to 0 (the
        min value for ``uint8`` arrays). Used if reference_im nodata value is a float.
        Ignored if reference_im nodata value is an int or if reference_im is not used.
        Take care when visualizing these masks, the nodata value may cause labels to not
        be visualized if nodata values are automatically masked by the software.

    Returns
    -------
    mask : :class:`numpy.array`
        A pixel mask with 0s for non-object pixels and `burn_value` at object
        pixels. `mask` dtype will coincide with `burn_value`.

    """
    # TODO: Refactor to remove some duplicated code here and in other mask fxns

    if out_file and not reference_im:
        raise ValueError(
            'If saving output to file, `reference_im` must be provided.')
    df = _check_df_load(df)

    if len(df) == 0: # for saving an empty mask.
        reference_im = _check_rasterio_im_load(reference_im)
        shape = reference_im.shape
        return np.zeros(shape=shape, dtype='uint8')

    if do_transform is None:
        # determine whether or not transform should be done
        do_transform = _check_do_transform(df, reference_im, affine_obj)

    df[geom_col] = df[geom_col].apply(_check_geom)  # load in geoms if wkt
    if not do_transform:
        affine_obj = Affine(1, 0, 0, 0, 1, 0)  # identity transform

    if reference_im:
        reference_im = _check_rasterio_im_load(reference_im)
        shape = reference_im.shape
        if do_transform:
            affine_obj = reference_im.transform

    # extract geometries and pair them with burn values

    if burn_field:
        if out_type == 'int':
            feature_list = list(zip(df[geom_col],
                                    df[burn_field].astype('uint8')))
        else:
            feature_list = list(zip(df[geom_col],
                                    df[burn_field].astype('float32')))
    else:
        feature_list = list(zip(df[geom_col], [burn_value]*len(df)))

    if out_type == 'int':
        output_arr = np.empty(shape=(shape[0], shape[1],
                                     len(feature_list)), dtype='uint8')
    else:
        output_arr = np.empty(shape=(shape[0], shape[1],
                                     len(feature_list)), dtype='float32')
    # initialize the output array

    for idx, feat in enumerate(feature_list):
        output_arr[:, :, idx] = features.rasterize([feat], out_shape=shape,
                                                   transform=affine_obj)

    if reference_im:
        reference_im = _check_rasterio_im_load(reference_im)
    try:
        bad_data_mask = (reference_im.read() == reference_im.nodata).any(axis=0) # take logical and along all dims so that all pixxels not -9999 across bands
    except AttributeError as ae:  # raise another, more verbose AttributeError
        raise AttributeError("A nodata value is not defined for the source image. Make sure the reference_im has a nodata value defined.") from ae
    if len(bad_data_mask.shape) > 2:
        bad_data_mask = np.dstack([bad_data_mask]*output_arr.shape[2])
        output_arr = np.where(bad_data_mask, 0, output_arr) # mask is broadcasted to filter labels where there are non-nan image values

    if out_file:
        meta = reference_im.meta.copy()
        meta.update(count=output_arr.shape[-1])
        if out_type == 'int':
            meta.update(dtype='uint8')
            if isinstance(meta['nodata'], float):
                meta.update(nodata=nodata_value)
        with rasterio.open(out_file, 'w', **meta) as dst:
            for c in range(1, 1 + output_arr.shape[-1]):
                dst.write(output_arr[:, :, c-1], indexes=c)
            dst.close()

    return output_arr

def geojsons_to_masks_and_fill_nodata(rtiler, vtiler, label_tile_dir, fill_value=0):
    """
    Converts tiled vectors to raster labels and fills nodata values in raster and vector tiles.

    This function must be run after a raster tiler and vector tiler have already been initialized
    and the `.tile()` method for each has been called to generate raster and vector tiles.
    Geojson labels are first converted to rasterized masks, then the labels are set to 0
    where the reference image, the corresponding image tile, has nodata values. Then, nodata
    areas in the image tile are filled  in place with the fill_value. Only works for rasterizing
    all geometries as a single category with a burn value of 1. See test_tiler_fill_nodata in
    tests/test_tile/test_tile.py for an example.

    Args
    -------
    rtiler : RasterTiler
        The RasterTiler that has had it's `.tile()` method called.
    vtiler : VectorTiler
        The VectorTiler that has had it's `.tile()` method called.
    label_tile_dir : str
        The folder path to save rasterized labels. This is created if it doesn't already exist.
    fill_value : str, optional
        The value to use to fill nodata values in images. Defaults to 0.

    Returns
    -------
    rasterized_label_paths : list
        A list of the paths to the rasterized instance masks.
    """
    rasterized_label_paths = []
    print("starting label mask generation")
    if not os.path.exists(label_tile_dir):
        os.mkdir(label_tile_dir)
    for img_tile, geojson_tile in tqdm(zip(sorted(rtiler.tile_paths), sorted(vtiler.tile_paths))):
        fid = os.path.basename(geojson_tile).split(".geojson")[0]
        rasterized_label_path = os.path.join(label_tile_dir, fid + ".tif")
        rasterized_label_paths.append(rasterized_label_path)
        gdf = gpd.read_file(geojson_tile)
        # gdf.crs = rtiler.raster_bounds_crs # add this because gdfs can't be saved with wkt crs
        arr = instance_mask(gdf, out_file=rasterized_label_path, reference_im=img_tile,
                                        geom_col='geometry', do_transform=None,
                                        out_type='int', burn_value=1, burn_field=None) # this saves the file, unless it is empty in which case we deal with it below.
        if not arr.any(): # in case no instances in a tile we save it with "empty" at the front of the basename
            with rasterio.open(img_tile) as reference_im:
                meta = reference_im.meta.copy()
                reference_im.close()
            meta.update(count=1)
            meta.update(dtype='uint8')
            if isinstance(meta['nodata'], float):
                meta.update(nodata=0)
            rasterized_label_path = os.path.join(label_tile_dir, "empty_" + fid + ".tif")
            with rasterio.open(rasterized_label_path, 'w', **meta) as dst:
                dst.write(np.expand_dims(arr, axis=0))
                dst.close()
    rtiler.fill_all_nodata(nodata_fill=fill_value)
    return rasterized_label_paths


############################################################################
# GEOTRANSFORMS FOR GEODF
############################################################################

def georegister_px_df(df, im_path=None, affine_obj=None, crs=None,
                      geom_col='geometry', precision=None, output_path=None):
    """Convert a dataframe of geometries in pixel coordinates to a geo CRS.

    Arguments
    ---------
    df : :class:`pandas.DataFrame`
        A :class:`pandas.DataFrame` with polygons in a column named
        ``"geometry"``.
    im_path : str, optional
        A filename or :class:`rasterio.DatasetReader` object containing an
        image that has the same bounds as the pixel coordinates in `df`. If
        not provided, `affine_obj` and `crs` must both be provided.
    affine_obj : `list` or :class:`affine.Affine`, optional
        An affine transformation to apply to `geom` in the form of an
        ``[a, b, d, e, xoff, yoff]`` list or an :class:`affine.Affine` object.
        Required if not using `raster_src`.
    crs : valid CRS `str`, `int`, or :class:`rasterio.crs.CRS` instance
        The coordinate reference system for the output GeoDataFrame as an EPSG
        code integer. Required if not providing a raster image to extract the
        information from.
    geom_col : str, optional
        The column containing geometry in `df`. If not provided, defaults to
        ``"geometry"``.
    precision : int, optional
        The decimal precision for output geometries. If not provided, the
        vertex locations won't be rounded.
    output_path : str, optional
        Path to save the resulting output to. If not provided, the object
        won't be saved to disk.

    """
    if im_path is not None:
        im = _check_rasterio_im_load(im_path)
        affine_obj = im.transform
        crs = im.crs
    else:
        if not affine_obj or not crs:
            raise ValueError('If an image path is not provided, '
                             'affine_obj and crs must be.')
    crs = _check_crs(crs)
    tmp_df = affine_transform_gdf(df, affine_obj, geom_col=geom_col,
                                  precision=precision)
#    result = gpd.GeoDataFrame(tmp_df, crs='epsg:' + str(crs.to_epsg()))
    result = gpd.GeoDataFrame(tmp_df)
    result = result.set_crs(crs, allow_override=True)

    if output_path is not None:
        if output_path.lower().endswith('json'):
            result.to_file(output_path, driver='GeoJSON')
        else:
            result.to_csv(output_path, index=False)

    return result

