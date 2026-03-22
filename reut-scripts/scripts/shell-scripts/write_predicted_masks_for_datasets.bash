#################################################################
#
# created for ramp project, August 2022
# Author: carolyn.johnston@dev.global
#
#################################################################

 
 

##################################################################

# Convenience script.
# Calculates predicted multichannel masks for each chip in each dataset in the list of datasets.
# Requires a trained model.

##################################################################
# USER TO SET

# model path to load
export MODEL_PATH=$RAMP_HOME/ramp-data/TRAIN/gimmosss/model-checkpts/20220809-220038/model_20220809-220038_010_0.964.tf

# model id for the trained model. 
# This will be used as an output directory name, for sorting output from different models.
export MODEL_ID=20220809-220038

# datasets to predict on (names of subdirectories of the input parent directory)
export DATASET_NAMES=(haiti ghana india malawi myanmar oman2 sierraleone south_sudan st_vincent)

# parent directory for all input chip datasets.
# chips should be in INPUT_PARENT_DIR/DATASET_NAME/chips
export INPUT_PARENT_DIR=$RAMP_HOME/ramp-data/TEST

# parent directory for all output multichannel mask datasets
# output will be written to OUTPUT_PARENT_DIR/DATASET_NAME/MODEL_ID/multimasks
export OUTPUT_PARENT_DIR=$RAMP_HOME/ramp-data/EVAL

# END USER-SET VARIABLES.
##################################################################

# scripts to execute
export SCRIPTS=$RAMP_HOME/ramp-staging/scripts
export PRED_MASKS_EXEC=$SCRIPTS/get_model_predictions.py

export PROD_DIR=$RAMP_HOME/ramp-data/PROD
export TRAIN_DIR=$RAMP_HOME/ramp-data/TRAIN
echo
echo 

for DATASET_NAME in "${DATASET_NAMES[@]}"; do

    echo ==========================================================
    echo
    echo
    echo PREDICTING VALIDATION MASKS FROM $DATASET_NAME

    export IN_DIR=$INPUT_PARENT_DIR/$DATASET_NAME/valchips
    export OUT_PATH=$OUTPUT_PARENT_DIR/$DATASET_NAME/$MODEL_ID/predicted_masks

    echo 
    echo python $PRED_MASKS_EXEC -mod $MODEL_PATH -idir $IN_DIR -odir $OUT_PATH
    python $PRED_MASKS_EXEC -mod $MODEL_PATH -idir $IN_DIR -odir $OUT_PATH
    
    echo 
    echo

echo DONE
done