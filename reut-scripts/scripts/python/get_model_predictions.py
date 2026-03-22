#################################################################
#
# created for ramp project, August 2022
# Author: carolyn.johnston@dev.global
#
# Modified March 2026:
#   - Removed tf.keras.backend.clear_session() every 100 batches
#     (was destroying TF graph and recompiling all CUDA kernels)
#   - Fixed NaN in normalization for all-zero nodata chips
#   - Fixed GPU config (removed contradictory set_memory_growth)
#   - Fixed model_dir.is_dir missing parentheses
#   - Skip all-zero chips entirely (saves GPU time on nodata tiles)
#
#################################################################

import os, sys
os.environ['TF_CUDNN_USE_AUTOTUNE'] = '0'
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

# ── GPU config (configurable via RAMP_GPU_MEMORY_MB env var) ────────
# Default: 11520 MiB = 75% of Tesla T4 15360 MiB
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    gpu_mem_mb = int(os.environ.get('RAMP_GPU_MEMORY_MB', '11520'))
    tf.config.set_logical_device_configuration(
        gpus[0],
        [tf.config.LogicalDeviceConfiguration(memory_limit=gpu_mem_mb)]
    )
# ────────────────────────────────────────────────────────────────────

def rasterio_get_image_tensor(open_rio_image):
    """Read image to float32 [0,1]. Returns None for all-zero (nodata) chips."""
    img = open_rio_image.read()
    # rasterio images are channel-first, we need channels last
    img = to_channels_last(img).astype("float32")
    # normalize the float image to [0,1]
    max_val = np.max(img)
    if max_val == 0:
        return None  # nodata chip — skip prediction entirely
    return img / max_val


# adding logging
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
    assert model_dir.is_dir(), f"model directory {model_dir} not readable"  # FIX: was is_dir (no parens)
    images_dir = Path(args.images_dir)
    assert images_dir.is_dir(), f'image data directory {images_dir} not readable'
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # compile==False: keras won't search for custom loss functions (not needed for prediction)
    model = tf.keras.models.load_model(model_dir, compile=False)

    chip_files = sorted(images_dir.glob('**/*.tif'))
    BATCH_SIZE = 16
    skipped = 0

    batches = [chip_files[i:i+BATCH_SIZE] for i in range(0, len(chip_files), BATCH_SIZE)]
    for batch_idx, batch in enumerate(tqdm(batches, desc="Predicting")):
        batch_tensors = []
        batch_profiles = []
        batch_masknames = []
        # Track which indices in the batch are valid (non-nodata)
        valid_indices = []
        # Track all profiles/names for writing (including nodata chips)
        all_profiles = []
        all_masknames = []

        for i, chip_file in enumerate(batch):
            bname = get_basename(str(chip_file))
            maskname = bname + ".pred.tif"
            with rio.open(chip_file, 'r') as src:
                dst_profile = src.profile
                dst_profile["count"] = 1
                if "photometric" in dst_profile and dst_profile["photometric"] == 'ycbcr':
                    del dst_profile["photometric"]
                all_profiles.append(dst_profile)
                all_masknames.append(maskname)

                tensor = rasterio_get_image_tensor(src)
                if tensor is not None:
                    batch_tensors.append(tensor)
                    batch_profiles.append(dst_profile)
                    batch_masknames.append(maskname)
                    valid_indices.append(i)
                else:
                    # Write blank mask for nodata chip
                    skipped += 1
                    mask_file = output_dir / maskname
                    blank = np.zeros((1, dst_profile["height"], dst_profile["width"]),
                                     dtype=dst_profile.get("dtype", "float32"))
                    with rio.open(mask_file, 'w', **dst_profile) as dst:
                        dst.write(blank)

        # Only run prediction if there are valid chips in this batch
        if batch_tensors:
            batch_array = np.stack(batch_tensors, axis=0)
            predictions = get_mask_from_prediction(model.predict(batch_array, verbose=0))

            for j in range(len(batch_tensors)):
                predicted = np.squeeze(predictions[j])
                mask_file = output_dir / batch_masknames[j]
                with rio.open(mask_file, 'w', **batch_profiles[j]) as dst:
                    dst.write(np.expand_dims(predicted, axis=0))

            del batch_tensors, batch_array, predictions

        # NOTE: We do NOT clear the TF session or reload the model.
        # The old code did this every 100 batches:
        #     tf.keras.backend.clear_session()
        #     model = tf.keras.models.load_model(model_dir, compile=False)
        # This destroys the entire TF computation graph and forces
        # recompilation of all CUDA kernels — adding ~30s overhead
        # each time. TF handles its own memory; this is not needed.

    log.info(f"Prediction complete. Skipped {skipped} nodata chips (written as blank masks).")


if __name__=="__main__":
    main()