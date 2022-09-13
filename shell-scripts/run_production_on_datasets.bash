#################################################################
#
# created for ramp project, August 2022
# Author: carolyn.johnston@dev.global
#
#################################################################

 
 
# Convenience script:

# Create fused predicted polygon geojson files for each dataset in a list of image chip datasets. 
# Requires a single trained model.

##############################################################

# USER TO SET

# directories in the INPUT_PARENT_DIR for which we want to generate predictions from image chips
export DATASET_NAMES=(ghana haiti india malawi myanmar oman2 sierraleone south_sudan st_vincent)

# path to the common parent of the input dataset folders
export INPUT_PARENT_DIR=/tf/ramp-data/TEST

# path to the common parent of the output dataset folders
export OUTPUT_PARENT_DIR=$RAMP_HOME/ramp-data/EVAL/TEST_SETS

# path to the model used to generate the predicted masks.
export MODEL_PATH=/tf/ramp-data/TRAIN/gimmosss/model-checkpts/best_gimmosss_model/model_20220809-220038_010_0.964.tf

# width of the building boundary class as defined in the training multichannel masks 
export BDRY_WIDTH_PIX=2

# Chips directory name.
# Are you running production on a validation chip set? If so switch to:
# CHIP_DIR=valchips
CHIP_DIR=chips 

##############################################################

# scripts to execute
export SCRIPTS=$RAMP_HOME/ramp-staging/scripts
export PRED_MASKS_EXEC=$SCRIPTS/get_model_predictions.py
export MASKS_TO_LABELS_EXEC=$SCRIPTS/get_labels_from_masks.py


echo
echo 

for DATASET_NAME in "${DATASET_NAMES[@]}"; do

    echo ==========================================================
    echo
    echo
    echo PROCESSING $DATASET_NAME
    echo

    # note: the fusion process creates a temporary directory containing predicted masks, reprojected to WGS84 lat long.
    # this directory is cleaned up at the end of this script.
    export TMP_REPROJECTED_MASK_PATH=$OUTPUT_PARENT_DIR/$DATASET_NAME/temp/predicted_masks_4326
    export INPUT_CHIPS_DIR=$INPUT_PARENT_DIR/$DATASET_NAME/$CHIP_DIR
        
    export OUTPUT_MASKS_DIR=$OUTPUT_PARENT_DIR/$DATASET_NAME/predicted
    export PREDICTION_JSON=$OUTPUT_PARENT_DIR/$DATASET_NAME/predicted_labels.geojson

    echo GENERATING MASK PREDICTIONS FROM MODEL
    echo 
    echo python $PRED_MASKS_EXEC -mod $MODEL_PATH -idir $INPUT_CHIPS_DIR -odir $OUTPUT_MASKS_DIR
    python $PRED_MASKS_EXEC -mod $MODEL_PATH -idir $INPUT_CHIPS_DIR -odir $OUTPUT_MASKS_DIR
    echo 
    echo

    echo CREATING FUSED OUTPUT BUILDING POLYGONS
    echo 
    echo python $MASKS_TO_LABELS_EXEC -in $OUTPUT_MASKS_DIR -out $PREDICTION_JSON -tmpdir $TMP_REPROJECTED_MASK_PATH -bwpix $BDRY_WIDTH_PIX
    python $MASKS_TO_LABELS_EXEC -in $OUTPUT_MASKS_DIR -out $PREDICTION_JSON -tmpdir $TMP_REPROJECTED_MASK_PATH -bwpix $BDRY_WIDTH_PIX

    echo CLEANING TEMP DIRECTORY $TMP_REPROJECTED_MASK_PATH
    rm -rf $TMP_REPROJECTED_MASK_PATH

done