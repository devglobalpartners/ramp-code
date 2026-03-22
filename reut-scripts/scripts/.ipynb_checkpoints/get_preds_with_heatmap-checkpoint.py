#################################################################
#
# Modified for RAMP project, August 2025
# Original Author: carolyn.johnston@dev.global
# Modifications: Added probability heatmap saving
#
#################################################################

import os, sys
from pathlib import Path
import rasterio as rio
import numpy as np
import argparse
import tensorflow as tf
from tqdm import tqdm

from ramp.utils.file_utils import get_basename
from ramp.utils.img_utils import to_channels_last, to_channels_first
from ramp.data_mgmt.display_data import get_mask_from_prediction


def rasterio_get_image_tensor(open_rio_image):
    img = open_rio_image.read()
    # rasterio images are channel-first, we need channels last
    img = to_channels_last(img).astype("float32")
    # normalize the float image to [0,1]
    return img / np.max(img)


# adding logging
import logging
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
log.addHandler(ch)


def main():

    parser = argparse.ArgumentParser(description=''' 
    Produces both predicted masks and probability heatmaps from input chips.

    Example: get_model_predictions.py -mod model_dir -idir chips -odir predicted_masks -hdir predicted_heatmaps
    ''', formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument('-mod', '--model_dir', type=str, required=True,
                        help="Path to directory containing tensorflow model")
    parser.add_argument('-idir', '--images_dir', type=str, required=True,
                        help='Input geotiff chips directory')
    parser.add_argument('-odir', '--output_dir', type=str, required=True,
                        help='Output geotiff predicted masks directory')
    parser.add_argument('-hdir', '--heatmap_dir', type=str, required=True,
                        help='Output geotiff probability heatmaps directory')

    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    assert model_dir.is_dir(), f"model directory {model_dir} not readable"
    images_dir = Path(args.images_dir)
    assert images_dir.is_dir(), f'image data directory {images_dir} not readable'
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    heatmap_dir = Path(args.heatmap_dir)
    heatmap_dir.mkdir(parents=True, exist_ok=True)

    # load model
    model = tf.keras.models.load_model(model_dir, compile=False)
    chip_files = sorted(images_dir.glob('**/*.tif'))

    for chip_file in tqdm(chip_files):
        bname = get_basename(str(chip_file))

        with rio.open(chip_file, 'r') as src:
            dst_profile = src.profile

            # ---- Prepare mask profile ----
            mask_profile = dst_profile.copy()
            mask_profile["count"] = 1
            mask_profile["dtype"] = "uint8"

            # ---- Prepare heatmap profile ----
            heatmap_profile = dst_profile.copy()
            heatmap_profile["count"] = 1
            heatmap_profile["dtype"] = "float32"

            if "photometric" in dst_profile and dst_profile["photometric"] == 'ycbcr':
                del mask_profile["photometric"]
                del heatmap_profile["photometric"]

            chip_tensor = rasterio_get_image_tensor(src)

            ##############################
            # get raw prediction for heatmap (squeezed)
            raw_pred = model.predict(np.expand_dims(chip_tensor, 0))
            raw_pred = np.squeeze(raw_pred, axis=0)   # (H, W, classes) or (H, W, 1)

            ##############################
            # ---- Save mask (use get_mask_from_prediction with batched input) ----
            maskname = bname + ".pred.tif"
            predicted_mask = get_mask_from_prediction(model.predict(np.expand_dims(chip_tensor, 0)))
            predicted_mask = np.squeeze(predicted_mask, axis=0)
            mask_file = output_dir / maskname
            with rio.open(mask_file, 'w', **mask_profile) as dst:
                dst.write(to_channels_first(predicted_mask))

            ##############################
            # ---- Save heatmap (probability for building class) ----
            heatmapname = bname + ".heatmap.tif"
            print("raw_pred shape:", raw_pred.shape, "min:", raw_pred.min(), "max:",
                  raw_pred.max())
            

            # explicitly grab "building" class (assumed index 1)
            if raw_pred.ndim == 3 and raw_pred.shape[-1] > 1:
                heatmap = raw_pred[..., 1]
            else:
                raise ValueError("Model output shape not as expected for multiclass")

            heatmap_file = heatmap_dir / heatmapname
            with rio.open(heatmap_file, 'w', **heatmap_profile) as dst:
                dst.write(heatmap.astype("float32"), 1)


if __name__ == "__main__":
    main()
