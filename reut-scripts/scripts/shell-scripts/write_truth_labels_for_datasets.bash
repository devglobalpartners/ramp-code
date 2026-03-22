#################################################################
#
# created for ramp project, August 2022
# Author: carolyn.johnston@dev.global
#
#################################################################

 
 
# Convenience script:
# Create truth label geojson files from a truth multichannel mask directory
# for each dataset in a list of datasets.


#####################################################################################
## USER TO SET

# list of datasets to create geojson label files for.
export DATASET_NAMES=(ghana haiti india malawi myanmar oman2 sierraleone south_sudan st_vincent)

# input multimasks will be in INPUT_PARENT_DIR/DATASET_NAME/multimasks
export INPUT_PARENT_DIR=$RAMP_HOME/ramp-data/TEST

# output geojson files will be in OUTPUT_PARENT_DIR/DATASET_NAME
export OUTPUT_PARENT_DIR=$RAMP_HOME/ramp-data/EVAL/TEST_SETS

# END USER SET VARIABLES
#####################################################################################

# scripts to execute
export SCRIPTS=$RAMP_HOME/ramp-staging/scripts
export MASKS_TO_LABELS_EXEC=$SCRIPTS/get_labels_from_masks.py


echo
echo 

for DATASET_NAME in "${DATASET_NAMES[@]}"; do

    echo ==========================================================
    echo
    echo
    echo PROCESSING $DATASET_NAME

    export IN_DIR=$INPUT_PARENT_DIR/$DATASET_NAME/multimasks
    export OUT_PATH=$OUTPUT_PARENT_DIR/$DATASET_NAME/${DATASET_NAME}_truth_labels.geojson
    export TMP_DIR=$OUTPUT_PARENT_DIR/$DATASET_NAME/temp

    echo 
    echo python $MASKS_TO_LABELS_EXEC -in $IN_DIR -out $OUT_PATH -tmpdir $TMP_DIR
    python $MASKS_TO_LABELS_EXEC -in $IN_DIR -out $OUT_PATH -tmpdir $TMP_DIR
    echo

    echo REMOVING TEMPORARY DIRECTORY $TMP_DIR 
    rm -rf $TMP_DIR

echo DONE
done