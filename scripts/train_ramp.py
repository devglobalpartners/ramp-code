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

# Note: this suppresses warning and other less urgent messages,
# and only allows errors to be printed.
# Comment this out if you are having mysterious problems, so you can see all messages.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

RAMP_HOME = os.environ["RAMP_HOME"]

from ramp.data_mgmt.data_generator import add_class_weights, training_batches_from_gtiff_dirs, test_batches_from_gtiff_dirs
from ramp.utils.misc_ramp_utils import log_experiment_to_file, get_num_files
from ramp.models.effunet_1 import get_effunet
from ramp.utils.model_utils import get_best_model_value_and_epoch
import ramp.utils.log_fields as lf
import json

# adding logging
# options for log levels: INFO, WARNING, DEBUG
import logging
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
log.addHandler(ch)

from ramp.training.augmentation_constructors import get_augmentation_fn 
from ramp.training import callback_constructors
from ramp.training import model_constructors
from ramp.training import optimizer_constructors
from ramp.training import metric_constructors
from ramp.training import loss_constructors

import segmentation_models as sm
sm.set_framework('tf.keras')

def main():

    parser = argparse.ArgumentParser(description='''
    This script performs ramp model training.

    Example: train_ramp.py -config train_config.json
    All configuration parameters are in the json file: train_config.json.
    Examples of configuration files are in the ramp codebase, in the 'experiments' directory.
    ''',formatter_class=argparse.RawTextHelpFormatter)
    
    parser.add_argument('-config', '--config', type=str, required=False, help="config file for training run")
    args = parser.parse_args()
    assert Path(args.config).is_file(), f"config file {args.config} not found"
    
    with open(args.config) as jf:
        cfg = json.load(jf)
    
    #uncomment to directly check gpu access
    tf.config.list_physical_devices('GPU') 

    # I added this code to prevent mysterious out of memory problems with the GPU.
    # see this webpage for explanation: 
    # https://stackoverflow.com/questions/43147983/could-not-create-cudnn-handle-cudnn-status-internal-error
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
    config = tf.config.experimental.set_memory_growth(physical_devices[0], True)
    config = tf.config.experimental.set_memory_growth(physical_devices[1], True)

    #### if this is just a test run to make sure everything works,
    #### you can turn off all logging to tensorboard and model checkpoints saving
    discard_experiment = False
    if "discard_experiment" in cfg:
         discard_experiment = cfg["discard_experiment"]

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
        get_metrics_fn_parms = cfg["metrics"]["metrics_fn_parms"]
        
        for get_mf_name, mf_parms in zip(get_metrics_fn_names, get_metrics_fn_parms):
            get_metric_fn = getattr(metric_constructors, get_mf_name)
            the_metrics.append(get_metric_fn(mf_parms))
            
    log.info(f"Accuracy metrics: {[fn.name for fn in the_metrics]}")

    #### construct optimizer ####
    get_optimizer_fn_name = cfg["optimizer"]["get_optimizer_fn_name"]
    get_optimizer_fn = getattr(optimizer_constructors, get_optimizer_fn_name)
    log.info(f"Optimizer: {get_optimizer_fn.__name__}")
    optimizer = get_optimizer_fn(cfg)

    # set random seed for reproducibility
    random_seed = int(cfg["random_seed"])
    random.seed(random_seed)

    #### construct model ####
    the_model = None
    if cfg["saved_model"]["use_saved_model"]:
        model_path = Path(RAMP_HOME) / cfg["saved_model"]["saved_model_path"]
        log.info(f"Model: importing saved model {str(model_path)}")
        the_model = tf.keras.models.load_model(model_path)

        assert the_model is not None, f"the saved model was not constructed: {model_path}"
        if not "save_optimizer_state" in cfg["saved_model"].keys() or not cfg["saved_model"]["save_optimizer_state"]:
            # compile the model
            # this will cause the model to lose its optimizer state
            the_model.compile(optimizer = optimizer, 
                loss=loss_fn,
                metrics = the_metrics)
    else:
        get_model_fn_name = cfg["model"]["get_model_fn_name"]
        get_model_fn = getattr(model_constructors, get_model_fn_name)
        log.info(f"Model constructor: {get_model_fn.__name__}")
        the_model = get_model_fn(cfg)

        assert the_model is not None, f"the model was not constructed: {model_path}"
        the_model.compile(optimizer = optimizer, 
            loss=loss_fn,
            metrics = the_metrics)

    
    #### define data directories ####
    train_img_dir = Path(RAMP_HOME) / cfg["datasets"]["train_img_dir"]
    train_mask_dir = Path(RAMP_HOME) / cfg["datasets"]["train_mask_dir"]
    val_img_dir = Path(RAMP_HOME) / cfg["datasets"]["val_img_dir"]
    val_mask_dir = Path(RAMP_HOME) / cfg["datasets"]["val_mask_dir"]

    #### get augmentation transform ####
    aug = None
    if cfg["augmentation"]["use_aug"]:
        log.info("Using augmentation")
        aug = get_augmentation_fn(cfg)

    ### SETUP

    n_epochs = cfg["num_epochs"]
    batch_size = cfg["batch_size"]
    input_img_shape = cfg["input_img_shape"]
    output_img_shape = cfg["output_img_shape"]

    n_training = get_num_files(train_img_dir, "*.tif")
    n_val = get_num_files(val_img_dir, "*.tif")
    steps_per_epoch = n_training // batch_size
    validation_steps = n_val // batch_size

    # add these back to the config 
    # in case they are needed by callbacks
    cfg["runtime"] = {}
    cfg["runtime"]["n_training"] = n_training
    cfg["runtime"]["n_val"] = n_val
    cfg["runtime"]["steps_per_epoch"] = steps_per_epoch
    cfg["runtime"]["validation_steps"] = validation_steps

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

    val_batches = test_batches_from_gtiff_dirs(val_img_dir, 
                                                 val_mask_dir, 
                                                 batch_size, 
                                                 input_img_shape, 
                                                 output_img_shape)

    assert val_batches is not None, "validation batches were not constructed"

    #### Construct all callbacks ####

    callbacks_list = []

    # get early stopping callback -- do not use with CLR
    if cfg["early_stopping"]["use_early_stopping"]:
        log.info("Using early stopping")
        callbacks_list.append(callback_constructors.get_early_stopping_callback_fn(cfg))

    # get cyclic learning scheduler callback
    if cfg["cyclic_learning_scheduler"]["use_clr"]:
        assert not cfg["early_stopping"]["use_early_stopping"], "cannot use early_stopping with cycling_learning_scheduler"
        get_clr_callback_fn_name = cfg["cyclic_learning_scheduler"]["get_clr_callback_fn_name"]
        get_clr_callback_fn = getattr(callback_constructors, get_clr_callback_fn_name)
        callbacks_list.append(get_clr_callback_fn(cfg))
        log.info(f"CLR callback constructor: {get_clr_callback_fn.__name__}")


    #### EXPERIMENT LOGGING AND MODEL CHECKPOINT CALLBACKS

    if not discard_experiment:

        # get model checkpoint callback
        if cfg["model_checkpts"]["use_model_checkpts"]:
            get_model_checkpt_callback_fn_name = cfg["model_checkpts"]["get_model_checkpt_callback_fn_name"]
            get_model_checkpt_callback_fn = getattr(callback_constructors, get_model_checkpt_callback_fn_name)
            callbacks_list.append(get_model_checkpt_callback_fn(cfg))
            log.info(f"model checkpoint callback constructor:{get_model_checkpt_callback_fn.__name__}")

        # get tensorboard callback
        if cfg["tensorboard"]["use_tb"]:
            get_tb_callback_fn_name = cfg["tensorboard"]["get_tb_callback_fn_name"]
            get_tb_callback_fn = getattr(callback_constructors, get_tb_callback_fn_name)
            callbacks_list.append(get_tb_callback_fn(cfg))
            log.info(f"tensorboard callback constructor: {get_tb_callback_fn.__name__}")

        # get tensorboard model prediction logging callback 
        if cfg["prediction_logging"]["use_prediction_logging"]:
            assert cfg["tensorboard"]["use_tb"], 'Tensorboard logging must be turned on to enable prediction logging'
            get_prediction_logging_fn_name = cfg["prediction_logging"]["get_prediction_logging_fn_name"]
            get_prediction_logging_fn = getattr(callback_constructors, get_prediction_logging_fn_name)
            callbacks_list.append(get_prediction_logging_fn(the_model, cfg))
            log.info(f"prediction logging callback constructor: {get_prediction_logging_fn.__name__}")

    # free up RAM
    keras.backend.clear_session()

    #### DO THE TRAINING ####

    if not "class_weighting" in cfg.keys() or not cfg["class_weighting"]["use_class_weighting"]:
        history = the_model.fit(train_batches, 
            epochs=n_epochs, 
            steps_per_epoch=steps_per_epoch, 
            validation_data=val_batches, 
            validation_steps=validation_steps,
            callbacks=callbacks_list)

    else:
        class_weights = cfg["class_weighting"]["class_weighting_parms"]
        log.info(f"Using loss function weighting with class parameters: {class_weights}")
        history = the_model.fit(train_batches.map(
            lambda chip, label: add_class_weights(chip, label, tf.constant(class_weights)) 
            ), 
            epochs=n_epochs, 
            steps_per_epoch=steps_per_epoch, 
            validation_data=val_batches, 
            validation_steps=validation_steps,
            callbacks=callbacks_list)

    # log values to a csv file
    
    if not discard_experiment and cfg["logging"]["log_experiment"]:
        exp_log_path = str(Path(RAMP_HOME)/cfg["logging"]["experiment_log_path"])

        # log some fields from this configuration
        # the fields must be uniquely named
        fields_to_log = cfg["logging"]["fields_to_log"]
        exp_log = dict()
        for field_key in fields_to_log:
            field_val = lf.locate_field(cfg, field_key)
            if not isinstance(field_val, dict):
                exp_log[field_key] = field_val
        the_argmax, best_val_acc = get_best_model_value_and_epoch(history)
        exp_log[lf.BEST_MODEL_VALUE] = best_val_acc
        exp_log[lf.BEST_MODEL_EPOCH] = the_argmax
        log_experiment_to_file(exp_log, exp_log_path)
        
if __name__=="__main__":
    main()