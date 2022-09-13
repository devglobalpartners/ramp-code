#################################################################
#
# created for ramp project, August 2022
# Author: carolyn.johnston@dev.global
#
#################################################################

 
 
import os, sys
from pathlib import Path
import numpy as np
import argparse
import tensorflow as tf
from tensorflow import keras
import datetime
import random
import matplotlib.pyplot as plt

# some options for log levels: INFO, WARNING, DEBUG
import logging
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
log.addHandler(ch)

# Note: this suppresses warning and other less urgent messages,
# and only allows errors to be printed.
# Comment this out if you are having mysterious problems, so you can see all the messages.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# ramp imports
RAMP_HOME = os.environ["RAMP_HOME"]

from ramp.data_mgmt.data_generator import training_batches_from_gtiff_dirs, test_batches_from_gtiff_dirs
from ramp.utils.misc_ramp_utils import log_experiment_to_file, get_num_files
from ramp.models.effunet_1 import get_effunet
from ramp.utils.model_utils import get_best_model_value_and_epoch
import ramp.utils.log_fields as lf
import json


from ramp.training.augmentation_constructors import get_augmentation_fn 
from ramp.training import model_constructors
from ramp.training import optimizer_constructors
from ramp.training import metric_constructors
from ramp.training import loss_constructors
from ramp.utils.lrfinder import LearningRateFinder

import segmentation_models as sm
sm.set_framework('tf.keras')
sm.framework()

def main():

    parser = argparse.ArgumentParser(description='''
    Utility for finding optimal learning rate for a given training configuration.
    Note: This utility uses ratefinder configuration files, 
    not regular training configuration files.
    A sample ratefinder configuration file is in the ramp codebase at: 
    experiments/20220805_himmosss_ratefinder.json.

    Example: find_learningrate.py -config ratefinder_config_file.json 
    ''',
    formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument('-config', '--config', type=str, required=False, help="configuration file for a ratefinder run")
    args = parser.parse_args()
    assert Path(args.config).is_file(), f"config file {args.config} not found"
    
    with open(args.config) as jf:
        cfg = json.load(jf)

    # set up runtime python code logging
    import logging
    logging.basicConfig(
                    stream = sys.stdout, 
                    level = logging.INFO)

    log = logging.getLogger()
    
    #uncomment to directly check gpu access
    tf.config.list_physical_devices('GPU') 

    # I added this code to prevent mysterious out of memory problems with the GPU.
    # see this webpage for explanation: 
    # https://stackoverflow.com/questions/43147983/could-not-create-cudnn-handle-cudnn-status-internal-error
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
    config = tf.config.experimental.set_memory_growth(physical_devices[0], True)
    config = tf.config.experimental.set_memory_growth(physical_devices[1], True)

    #### add timestamp to configuration ####
    cfg["timestamp"] = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    #### construct loss function ####
    get_loss_fn_name = cfg["loss"]["get_loss_fn_name"]
    # get_loss_fn_parms = cfg["loss"]["get_loss_fn_parms"]
    get_loss_fn = getattr(loss_constructors,get_loss_fn_name)
    log.info(f"Loss function: {get_loss_fn.__name__}")
    loss_fn = get_loss_fn(cfg)

    ### construct metrics
    the_metrics = []
    if cfg["metrics"]["use_metrics"]:
        get_metrics_fn_names = cfg["metrics"]["get_metrics_fn_names"]
        for get_mf_name in get_metrics_fn_names:
            get_metric_fn = getattr(metric_constructors, get_mf_name)
            the_metrics.append(get_metric_fn(cfg))
    log.info(f"Accuracy metrics: {[fn.name for fn in the_metrics]}")

    #### construct optimizer ####
    get_optimizer_fn_name = cfg["optimizer"]["get_optimizer_fn_name"]
    get_optimizer_fn = getattr(optimizer_constructors, get_optimizer_fn_name)
    log.info(f"Optimizer: {get_optimizer_fn.__name__}")
    optimizer = get_optimizer_fn(cfg)

    #### construct model ####
    the_model = None
    if cfg["saved_model"]["use_saved_model"]:
        raise NotImplementedError("starting from a saved model is not currently implemented")
    else:
        get_model_fn_name = cfg["model"]["get_model_fn_name"]
        get_model_fn = getattr(model_constructors, get_model_fn_name)
        the_model = get_model_fn(cfg)
        log.info(f"Model constructor: {get_model_fn.__name__}")
    assert the_model is not None, "the model was not constructed"
    
    # compile the model
    random_seed = cfg["random_seed"]
    the_model.compile(optimizer = optimizer, 
                        loss=loss_fn,
                        metrics = the_metrics)

    #### define data directories ####
    # You will only need the training data directories to find the optimal learning rate.
    train_img_dir = Path(RAMP_HOME) / cfg["datasets"]["train_img_dir"]
    train_mask_dir = Path(RAMP_HOME) / cfg["datasets"]["train_mask_dir"]
    # val_img_dir = Path(RAMP_HOME) / cfg["datasets"]["val_img_dir"]
    # val_mask_dir = Path(RAMP_HOME) / cfg["datasets"]["val_mask_dir"]

    #### get augmentation transform ####
    aug = None
    if cfg["augmentation"]["use_aug"]:
        aug = get_augmentation_fn(cfg)

    ### SETUP the learning rate finder.

    batch_size = cfg["batch_size"]
    input_img_shape = cfg["input_img_shape"]
    output_img_shape = cfg["output_img_shape"]

    n_training = get_num_files(train_img_dir, "*.tif")
    steps_per_epoch = n_training // batch_size
 
    # add these back to the config 
    # in case they are needed by callbacks
    cfg["runtime"] = {}
    cfg["runtime"]["n_training"] = n_training
    cfg["runtime"]["steps_per_epoch"] = steps_per_epoch
    
    ### set up training batches with or without augmentation
    train_batches = None
    if aug is not None:
        train_batches = training_batches_from_gtiff_dirs(train_img_dir, 
                                                 train_mask_dir, 
                                                 batch_size, 
                                                 input_img_shape, 
                                                 output_img_shape, 
                                                 transforms = aug)
    else:
        train_batches = training_batches_from_gtiff_dirs(train_img_dir, 
                                                 train_mask_dir, 
                                                 batch_size, 
                                                 input_img_shape, 
                                                 output_img_shape)

    assert train_batches is not None, "training batches were not constructed"

    # free up RAM
    keras.backend.clear_session()

    learningrate_finder = LearningRateFinder(the_model, cfg)
    learningrate_finder.find(train_batches)
    learningrate_finder.plot_loss()

    lrfind_plot_path = Path(learningrate_finder.lrfind_plot_path)
    lrfind_plot_path.parent.mkdir(exist_ok=True, parents=True)
    log.info(f"Writing the output learning rate plot to {learningrate_finder.lrfind_plot_path}")
    plt.savefig(learningrate_finder.lrfind_plot_path)
 
        
if __name__=="__main__":
    main()