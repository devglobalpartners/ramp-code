#################################################################
#
# created for ramp project, August 2022
# Author: carolyn.johnston@dev.global
#
#################################################################

import glob, os
from unittest import result

def locate_field(e, field_name):
    item = e.get(field_name, None)
    if item is not None:
        return item
    else:
        for key in e.keys():
            item = e[key]
            if isinstance(item, dict):
                # print(f"{key} points to a dictionary")
                result = locate_field(item, field_name)
                if result is not None:
                    return result
    return None

# Names of standard hyperparameters
# for use in experiment logs

EXP_NOTES = 'EXP_NOTES'
N_EPOCHS = 'N_EPOCHS'
BATCH_SIZE = 'BATCH_SIZE'
LEARNING_RATE = 'LEARNING_RATE'
LOSS_FN = 'LOSS_FN'
OPTIMIZER = 'OPTIMIZER'
TB_LOGDIR = 'TB_LOGDIR'
TIMESTAMP = 'TIMESTAMP'
TRAINING_IMG_DIR = 'TRAINING_IMG_DIR'
TRAINING_MASK_DIR = 'TRAINING_MASK_DIR'
VAL_IMG_DIR = 'VAL_IMG_DIR'
VAL_MASK_DIR = 'VAL_MASK_DIR'
OUTPUT_IMG_SIZE = 'OUTPUT_IMG_SIZE'
MODEL_FN = 'MODEL_FN'
RANDOM_SEED = 'RANDOM_SEED'
N_CLASSES = 'N_CLASSES'
BEST_MODEL_VALUE = 'BEST_MODEL_VALUE'
BEST_MODEL_EPOCH = 'BEST_MODEL_EPOCH'
KEEPER = 'KEEPER'
ENCODER_FREEZE = 'ENCODER_FREEZE'