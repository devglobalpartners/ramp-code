#################################################################
#
# created for ramp project, August 2022
# Author: carolyn.johnston@dev.global
#
#################################################################

 
 
# adding logging
# some options for log levels: INFO, WARNING, DEBUG
import logging
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
log.addHandler(ch)

# get rid of geopandas FutureWarning
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import sys, os
import numpy as np
import argparse
from pathlib import Path
import pandas as pd
import geopandas as gpd
import rasterio as rio
from rasterio import features
from tensorflow.keras.models import load_model as tf_load_model
from tqdm import tqdm

RAMP_HOME = os.environ["RAMP_HOME"]

from ramp.utils.file_utils import get_matching_pairs, get_basename
from ramp.utils.img_utils import gdal_get_image_tensor, nodata_count, get_nodata_mask_from_image, to_channels_last, to_channels_first
from ramp.data_mgmt.data_generator import test_batches_from_gtiff_dirs_with_names
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.metrics import SparseCategoricalAccuracy


def main():
    parser = argparse.ArgumentParser(description='''
    For a given model and chip-label dataset, creates a csv file listing sparse categorical loss & accuracy per chip.
    This currently only supports sparse categorical accuracy: in the future it will support arbitrary loss functions.
    Used for finding chip-label pairs with extreme quality problems (corrupt labels, missing labels)
    Input and output chip width and height should match the values defined when the model was trained:
        have a look in the ramp training config file. 
        
    Example: get_dataset_loss_statistics.py -m model_directory -idir chips -mdir multimasks -csv loss_values.csv    
    ''',
    formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-m', '--model_dir', type=str, required=True, help="Directory containing saved segmentation model" )
    parser.add_argument('-idir', '--image_dir', type = str, required = True, help="Directory containing image chip geotiffs")
    parser.add_argument('-mdir', '--mask_dir', type =str, required = True,  help='Directory containing training masks')
    parser.add_argument('-csv', '--csv', type=str, required=True, help='path to output csv file of training statistics' )

    # optional
    parser.add_argument('-iw', '--input_width', type=int, required=False, default=256, help="input chip width to training data generator (default 256)")
    parser.add_argument('-ih', '--input_height', type=int, required=False, default=256, help="input chip height to training data generator (default 256)")
    parser.add_argument('-ow', '--output_width', type=int, required=False, default=256, help="output chip width to training data generator (default 256)")
    parser.add_argument('-oh', '--output_height', type=int, required=False, default=256, help="output chip height to training data generator (default 256)")
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    assert model_dir.is_dir(), f"model directory {str(model_dir)} not readable"
    log.info(f"Loading model: {model_dir}")

    model = tf_load_model(model_dir)

    imdir = Path(args.image_dir)
    assert imdir.is_dir(), f"image directory {str(imdir)} not readable"
    mdir = Path(args.mask_dir)
    assert mdir.is_dir(), f"mask directory {str(ldir)} not readable"
    outcsv = Path(args.csv)    
    input_image_size = (args.input_height, args.input_width )
    output_image_size = (args.output_height, args.output_width)
    batch_size = 1

    # define loss and accuracy
    scce = SparseCategoricalCrossentropy()
    scacc = SparseCategoricalAccuracy()
    
    # lists that will be populated for final training set stats output
    accuracies = []
    losses = []
    imfiles = []
    mfiles = []

    test_batches = test_batches_from_gtiff_dirs_with_names(str(imdir), 
                                            str(mdir), 
                                            batch_size, 
                                            input_image_size, 
                                            output_image_size)

    for test_batch in tqdm(test_batches):

        image = test_batch[0][0]
        image_filename = test_batch[0][1]
        true_mask = test_batch[1][0]
        true_mask_filename = test_batch[1][1]

        prediction = model.predict(image)
        loss = scce(true_mask, prediction)
        acc = scacc(true_mask, prediction)

        # this is the filename wrangling necessary to get the string from the TF object, 
        # recode it from byte to utf-8, 
        # and remove its full path prefix.
        byte_filepath = image_filename.numpy()[0]
        filename = Path(byte_filepath.decode('UTF-8')).name   
        imfiles.append(filename)
        
        byte_filepath = true_mask_filename.numpy()[0]
        filename = Path(byte_filepath.decode('UTF-8')).name   
        mfiles.append(filename)
        
        losses.append(loss.numpy())
        accuracies.append(acc.numpy())

    # Done looping over chip-label pairs: write the statistics to CSV

    stats = pd.DataFrame(
        {
            "image_file":imfiles,
            "mask_file":mfiles,
            "loss":losses,
            "accuracy":accuracies
        }
    )
    log.info(f"Writing csv file: {str(outcsv)}")
    stats.to_csv(outcsv)


if __name__=="__main__":
    main()