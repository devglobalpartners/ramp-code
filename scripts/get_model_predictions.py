#################################################################
#
# created for ramp project, August 2022
# Author: carolyn.johnston@dev.global
#
#################################################################

 
 
import os, sys
from pathlib import Path
import rasterio as rio
from rasterio.features import shapes
import numpy as np
import argparse
import tensorflow as tf
from tqdm import tqdm
import geopandas as gpd, pandas as pd

from ramp.utils.file_utils import get_basename
from ramp.utils.img_utils import to_channels_last, to_channels_first
from ramp.data_mgmt.display_data import get_mask_from_prediction

def rasterio_get_image_tensor(open_rio_image):
    img = open_rio_image.read()
    # rasterio images are channel-first, we need channels last
    img = to_channels_last(img).astype("float32")
      # normalize the float image to [0,1]
    return img/np.max(img)


# adding logging
# some options for log levels: INFO, WARNING, DEBUG
import logging
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
log.addHandler(ch)

def main():

    parser = argparse.ArgumentParser(description=''' 
    Produces output predicted 4-channel masks from input chips.

    Example: get_model_predictions.py -mod model_dir -idir chips -odir predicted_masks. 
    ''',formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-mod', '--model_dir', type=str, required=True, help="Path to directory containing tensorflow model")
    parser.add_argument('-idir', '--images_dir', type=str, required = True, help='Input geotiff chips directory')
    parser.add_argument('-odir', '--output_dir', type=str, required = True, help='Output geotiff predicted masks directory')

    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    assert model_dir.is_dir, f"model directory {model_dir} not readable"
    images_dir = Path(args.images_dir)
    assert images_dir.is_dir(), f'image data directory {images_dir} not readable'
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # compile==False causes keras to not search for custom loss functions, which are not needed
    # if you're only doing prediction. 
    # See https://stackoverflow.com/questions/48373845/loading-model-with-custom-loss-keras
    model = tf.keras.models.load_model(model_dir, compile=False)
    chip_files = sorted(images_dir.glob('**/*.tif'))

    for chip_file in tqdm(chip_files):
        bname = get_basename(str(chip_file))
        maskname = bname + ".pred.tif"
        with rio.open(chip_file, 'r') as src:
            dst_profile = src.profile
            dst_profile["count"] = 1
            chip_tensor = rasterio_get_image_tensor(src) 

            # get the prediction and convert to a uint8 single-channel mask. 
            # We need to call np.expand_dims to add a batch dimension.
            predicted = get_mask_from_prediction(model.predict(np.expand_dims(chip_tensor,0)))

            # get rid of batch dimension, leaving shape (height, width, 1)
            predicted = np.squeeze(predicted, axis=0)
            mask_file = output_dir / maskname
            with rio.open(mask_file, 'w', **dst_profile) as dst:
                dst.write(to_channels_first(predicted))


if __name__=="__main__":
    main()
        

    