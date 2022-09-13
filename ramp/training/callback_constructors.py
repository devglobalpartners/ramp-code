#################################################################
#
# created for ramp project, August 2022
# Author: carolyn.johnston@dev.global
#
#################################################################

# adding logging
import logging
log = logging.getLogger(__name__)
log.addHandler(logging.NullHandler())
import os, sys
import tensorflow as tf
import keras
import keras.backend as K
import datetime
from pathlib import Path
import numpy as np

RAMP_HOME = os.environ["RAMP_HOME"]

import tensorflow.summary as tf_summary



from ..data_mgmt.clr_callback import CyclicLR
from . import metric_constructors

################################## HELPER FUNCTIONS

def get_tb_log_dir(cfg):
    timestamp = cfg["timestamp"]
    log_dir = str(Path(RAMP_HOME)/ cfg["tensorboard"]["tb_logs_dir"] / f"log-{timestamp}")
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    return log_dir

def get_accuracy_name(cfg):
    # get the accuracy name from the first metric listed
    assert cfg["metrics"]["use_metrics"], "metrics are not defined"
    get_metrics_fn_names = cfg["metrics"]["get_metrics_fn_names"]
    get_metrics_fn_name = get_metrics_fn_names[0]
    get_metrics_fn = getattr(metric_constructors, get_metrics_fn_name)
    metrics_fn = get_metrics_fn(cfg)
    return metrics_fn.name

################################## CALLBACKS
def get_tb_callback_fn(cfg):
    '''
    returns tensorboard logging callback function
    '''
    log_dir = get_tb_log_dir(cfg)
    tb_callback_parms = cfg["tensorboard"]["tb_callback_parms"]
    return tf.keras.callbacks.TensorBoard(log_dir, **tb_callback_parms)

def get_model_checkpt_callback_fn(cfg):
    '''
    return model checkpt callback fn
    '''
    accuracy_name = get_accuracy_name(cfg)
    val_acc_name = f"val_{accuracy_name}"

    timestamp = cfg["timestamp"]
    models_dir = str(Path(RAMP_HOME)/ cfg["model_checkpts"]["model_checkpts_dir"]/timestamp)
    Path(models_dir).mkdir(parents=True, exist_ok=True)
    model_checkpt_callback_parms = cfg["model_checkpts"]["model_checkpt_callback_parms"]
    Path(models_dir).mkdir(parents=True, exist_ok=True)
    
    # Note to self: '{{' in an f-string escapes the curly braces.
    model_filename = os.path.join(models_dir, f'model_{timestamp}_'+ f'{{epoch:03d}}_{{{val_acc_name}:.3f}}.tf')

    return keras.callbacks.ModelCheckpoint(model_filename, 
                                                monitor = val_acc_name,
                                                **model_checkpt_callback_parms
                                                )

def get_prediction_logging_fn(model, cfg):

    log_dir = get_tb_log_dir(cfg)
    
    #### define data directories ####
    val_img_dir = Path(RAMP_HOME) / cfg["datasets"]["val_img_dir"]
    val_mask_dir = Path(RAMP_HOME) / cfg["datasets"]["val_mask_dir"]
    batch_size = cfg["batch_size"]
    input_img_shape = cfg["input_img_shape"]
    output_img_shape = cfg["output_img_shape"]

    log.debug(f"logging predicted masks to logging directory: {log_dir}")
    log.debug(f"model class: {type(model)}")
    log.debug(f"test image directory: {val_img_dir}")
    log.debug(f"test mask directory: {val_mask_dir}")
    log.debug(f"batch_size: {batch_size}")
    log.debug(f"input image shape: {input_img_shape}")
    log.debug(f"output image shape: {output_img_shape}")

    def log_predictions(epoch, logs):
        '''
        callback function for logging predictions after every epoch.
        '''

        from ..data_mgmt.data_generator import test_batches_from_gtiff_dirs_with_names
        from ..data_mgmt.display_data import get_mask_from_prediction, plot_to_image, get_img_mask_pred_batch_figure_with_id
        
        # define a folder for prediction logging
        file_writer_preds = tf_summary.create_file_writer(log_dir + '/predictions')
        
        # Generate another round of test batches.
        val_batches = test_batches_from_gtiff_dirs_with_names(
                                                    val_img_dir, 
                                                    val_mask_dir, 
                                                    batch_size, 
                                                    input_img_shape, 
                                                    output_img_shape)

        # take a single batch from the data generator.
        # Note it has the same batch_size, height, and width, as the original image
        batch = next(val_batches.take(1).as_numpy_iterator())
        image_batch_tensor = batch[0][0]
        list_of_chip_names = batch[0][1]
        mask_batch_tensor = batch[1][0]
        prediction = model.predict(image_batch_tensor)

        # make a mask from the prediction.
        predmask_batch_tensor = get_mask_from_prediction(prediction)

        # display the result of training.
        img_to_log = plot_to_image(
            get_img_mask_pred_batch_figure_with_id(
                list_of_chip_names, 
                image_batch_tensor, 
                mask_batch_tensor, 
                predmask_batch_tensor)
                )

        # Log the confusion matrix as an image summary.
        with file_writer_preds.as_default():
            tf_summary.image("Test Batch Predictions", img_to_log, step=epoch)
            
    return log_predictions
    
def get_pred_logging_callback_fn(model, cfg):
    return keras.callbacks.LambdaCallback(
        on_epoch_end=get_prediction_logging_fn(model, cfg)
    )

def get_clr_callback_fn(cfg):
    n_training = cfg["runtime"]["n_training"]
    batch_size = cfg["batch_size"]
    
    # get clr parameters
    parms = cfg["cyclic_learning_scheduler"]["clr_callback_parms"]
    mode = parms["mode"]
    step_size = parms["stepsize"]*(n_training//batch_size)
    max_lr = parms["max_lr"]
    base_lr = parms["base_lr"]
    
    # plot clr schedule
    plot_dir = Path(RAMP_HOME) / cfg["cyclic_learning_scheduler"]["clr_plot_dir"] / cfg["timestamp"]
    plot_dir.mkdir(parents=True, exist_ok=True)
    plot_png = str( plot_dir / "clr_schedule_plot.png")
    return CyclicLR(mode=mode, base_lr=base_lr, max_lr = max_lr, step_size = step_size, plot_path= plot_png)

def get_early_stopping_callback_fn(cfg):
    estop_parms = cfg["early_stopping"]["early_stopping_parms"]
    return tf.keras.callbacks.EarlyStopping(**estop_parms)