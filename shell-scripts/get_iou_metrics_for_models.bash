#################################################################
#
# created for ramp project, August 2022
# Author: carolyn.johnston@dev.global
#
#################################################################

 
 
# CONVENIENCE SCRIPT
# CALCULATE RECALL, PRECISION, AND F1 USING IoU@0.5 (THE SPACENET METRIC)
# OVER A SINGLE CHIPS/LABELS TRUTH DATASET
# FOR EACH MODEL IN A PARENT DIRECTORY OF MODELS.
##############################################################

# USER TO SET
# UNIQUE ID FOR THE SINGLE TEST DATASET
export DATASET_NAME=dhaka_barishal_all_val

# DIRECTORIES CONTAINING MATCHING CHIPS AND LABELS FOR THE SINGLE TEST DATASET
export TRUTH_CHIPS_DIR=$RAMP_HOME/ramp-data/TRAIN/dhaka_barishal/valchips
export TRUTH_LABELS_FILE=$RAMP_HOME/ramp-data/EVAL/dhakabar/all_val_sample/dhakabar_val_truth_labels.json

# DIRECTORY CONTAINING ALL MODELS WHOSE ACCURACY IS TO BE CALCULATED
export PARENT_MODELS_DIR=$RAMP_HOME/ramp-data/EVAL/dhakabar/all_val_sample/test_models

# PARENT DIRECTORY TO CONTAIN ALL OUTPUT PREDICTION DATA FOR THE SINGLE TEST DATASET AND EACH MODEL
# all data generated from this process is contained in PARENT_OUTPUT_DIR/DATASET_NAME/MODEL_NAME
export PARENT_OUTPUT_DIR=$RAMP_HOME/ramp-data/EVAL

# minimum building size (area in m^2) to analyze
# zero if no minimum
export MIN_BLDG_AREA=5.0

# IOU THRESHOLD FOR MATCHING DETECTED POLYGONS TO TRUTH POLYGONS IN THE ACCURACY CALCULATION
# 0.5 IS THE STANDARD FOR THE SPACENET ACCURACY METRIC
export IOU_THRESH=0.5

# WIDTH OF BOUNDARY PIXELS OF THE MASKS USED IN TRAINING THE MODELS
export BDRY_WIDTH_PIX=2

# file for logging accuracy results
export LOG_FILE=$RAMP_HOME/ramp-data/EVAL/all_iou_results.csv

# A comment (no spaces allowed) to include with logging
export COMMENT=dhakabar_top_models_vs_full_val_data
# END USER-SET VARIABLES
##############################################################

# scripts to execute
export SCRIPTS=$RAMP_HOME/ramp-staging/scripts
export PRED_MASKS_EXEC=$SCRIPTS/get_model_predictions.py
export MASKS_TO_LABELS_EXEC=$SCRIPTS/get_labels_from_masks.py
export IOU_EXEC=$SCRIPTS/calculate_accuracy_iou.py

# get list of all model directories under the parent model directory
export MODEL_NAMES=($(ls $PARENT_MODELS_DIR))

for MODEL_NAME in "${MODEL_NAMES[@]}"; do


    echo =========================== PROCESSING MODEL: $MODEL_NAME ===============================
    echo
    echo

    export MODEL_PATH=$PARENT_MODELS_DIR/$MODEL_NAME

    export TMP_MASK_PATH=$PARENT_OUTPUT_DIR/$DATASET_NAME/$MODEL_NAME/predicted_masks
    export TMP_REPROJECTED_MASK_PATH=$PARENT_OUTPUT_DIR/$DATASET_NAME/$MODEL_NAME/predicted_masks_4326
    export TEST_LABELS_FILE=$PARENT_OUTPUT_DIR/$DATASET_NAME/$MODEL_NAME/labels_${MODEL_NAME}.geojson
    
    echo PREDICTING MASKS USING MODEL: $MODEL_NAME
    echo 
    echo python $PRED_MASKS_EXEC -mod $MODEL_PATH -idir $TRUTH_CHIPS_DIR -odir $TMP_MASK_PATH
    python $PRED_MASKS_EXEC -mod $MODEL_PATH -idir $TRUTH_CHIPS_DIR -odir $TMP_MASK_PATH
    echo 
    echo
    
    echo CREATING FUSED OUTPUT DATASET FROM $MODEL_NAME
    echo 
    echo python $MASKS_TO_LABELS_EXEC -in $TMP_MASK_PATH -out $TEST_LABELS_FILE -tmpdir $TMP_REPROJECTED_MASK_PATH -bwpix $BDRY_WIDTH_PIX
    python $MASKS_TO_LABELS_EXEC -in $TMP_MASK_PATH -out $TEST_LABELS_FILE -tmpdir $TMP_REPROJECTED_MASK_PATH -bwpix $BDRY_WIDTH_PIX
    echo 
    echo

    echo CALCULATING SPACENET METRIC VALUES FOR $MODEL_NAME
    echo python $IOU_EXEC -truth $TRUTH_LABELS_FILE -test $TEST_LABELS_FILE -f_m2 $MIN_BLDG_AREA -iou $IOU_THRESH -log $LOG_FILE -dset $DATASET_NAME -mid $MODEL_NAME -note $COMMENT
    python $IOU_EXEC -truth $TRUTH_LABELS_FILE -test $TEST_LABELS_FILE -f_m2 $MIN_BLDG_AREA -iou $IOU_THRESH -log $LOG_FILE -dset $DATASET_NAME -mid $MODEL_NAME -note $COMMENT
    echo 
    echo

    echo CLEANING TEMP DIRECTORIES
    echo rm $TMP_MASK_PATH/\* $TMP_REPROJECTED_MASK_PATH/\*
    rm $TMP_MASK_PATH/* $TMP_REPROJECTED_MASK_PATH/*
    echo
done

echo DONE