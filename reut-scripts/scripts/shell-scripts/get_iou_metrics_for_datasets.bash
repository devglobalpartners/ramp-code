#################################################################
#
# created for ramp project, August 2022
# Author: carolyn.johnston@dev.global
#
#################################################################

 
 

# CALCULATES RECALL, PRECISION, AND F1 USING IoU@0.5 (THE SPACENET METRIC)
# FOR EACH TRUTH GEOJSON FILE/PREDICTED GEOJSON FILE IN A LIST OF DATASETS,
# AND LOG THEM TO A COMMON CSV LOG FILE.

##############################################################
# USER TO SET

# NAMES OF ALL DATASETS TO CALCULATE ACCURACY FOR
export DATASET_NAMES=(haiti india malawi myanmar oman2 sierraleone south_sudan st_vincent)

# COMMON PARENT DIRECTORY UNDER WHICH PREDICTED AND TRUTH GEOJSONS ARE LOCATED FOR EACH DATASET
# TRUTH LABEL FILES SHOULD HAVE THE PATH: PARENT_DATA_DIR/DATASET_NAME/DATASET_NAME_truth_labels.geojson
# PREDICTED LABEL FILES SHOULD HAVE THE PATH: PARENT_DATA_DIR/DATASET_NAME/predicted_labels.geojson
# THESE ARE THE FILES THAT ARE CREATED BY write_truth_files.bash AND run_production_on_datasets.bash
export PARENT_DATA_DIR=$RAMP_HOME/ramp-data/EVAL/TEST_SETS

# minimum building size (area in m^2) to analyze
# comment out if no minimum
export FILTER_FLAG='-f_m2 5.0'

# VARIABLES DEFINED BELOW RELATE TO RESULT LOGGING
# csv file to log results in
export LOGFILE=$RAMP_HOME/ramp-data/EVAL/all_iou_results.csv

# comment (no spaces allowed) to include with logging - optional
export COMMENT=best_bootstrap_model_evaluated_on_test_sets

# model id to include with logging - optional
export MODEL_ID=model_20220809-220038_010_0.964

# 'test' or 'val' suffix to include in logging - optional
export DATASET_ID_SUFFIX=_test
# export DATASET_ID_SUFFIX=_val

# END USER SET VARIABLES
##############################################################
# scripts to execute
export SCRIPTS=$RAMP_HOME/ramp-staging/scripts
export IOU_EXEC=$SCRIPTS/calculate_accuracy_iou.py

echo
echo 

for DATASET_NAME in "${DATASET_NAMES[@]}"; do

    echo ==========================================================
    echo
    echo
    echo PROCESSING $DATASET_NAME

    export TRUTH_PATH=$PARENT_DATA_DIR/$DATASET_NAME/${DATASET_NAME}_truth_labels.geojson
    export TEST_PATH=$PARENT_DATA_DIR/$DATASET_NAME/predicted_labels.geojson
    export DATASET_ID=${DATASET_NAME}${DATASET_ID_SUFFIX}
    
    echo 
    echo python $IOU_EXEC -truth $TRUTH_PATH -test $TEST_PATH $FILTER_FLAG -log $LOGFILE -dset $DATASET_ID -mid $MODEL_ID -note $COMMENT
    python $IOU_EXEC -truth $TRUTH_PATH -test $TEST_PATH $FILTER_FLAG -log $LOGFILE -dset $DATASET_ID -mid $MODEL_ID -note $COMMENT

    echo 
    echo

echo DONE
done
