#################################################################
#
# created for ramp project, August 2022
# Author: carolyn.johnston@dev.global
#
#################################################################

 
 
# Convenience script to create a set of test datasets.

# Each PREP_BASE directory should contain a file named PREP_BASE/test_chips.csv, 
# which contains a list of the names of the chip files in PREP_BASE/chips 
# that are selected to be in the test set.
# This list is usually created manually in QGIS or another GIS tool.

# This script will copy or move the test files to the TEST_BASE directories. 
###################################################
## USER TO SET

# set name of parent directory for all input chip/label dataset directories
# chips are in INPUT_PREP_DIR/DATASET_NAME/chips
# labels are in INPUT_PREP_DIR/DATASET_NAME/labels
# this should not usually be changed.
export INPUT_PREP_DIR=$RAMP_HOME/ramp-data/PREP

# output dataset parent directory for all test data.
# this should not usually be changed.
export OUTPUT_TEST_DIR=$RAMP_HOME/ramp-data/TEST

# list of directories containing chips and labels in the INPUT_PREP_DIR directory.
export DATASET_NAMES=(coxbzr2)

# comment this line if you want the data files 
# to be copied from the source directory, not moved
export MOVE_FILES=-mv

###################################################

# script to execute
export SCRIPTS=$RAMP_HOME/ramp-staging/scripts
export MOVE_EXEC=$SCRIPTS/move_chips_from_csv.py


#for DATASET_NAME in $DATASET_NAMES; do
for DATASET_NAME in "${DATASET_NAMES[@]}"; do

    echo 
    echo PROCESSING $DATASET_NAME
    echo 

    # parent input directory for this dataset
    export PREP_BASE=$INPUT_PREP_DIR/$DATASET_NAME

    # output directory for dataset -- will be created if it doesn't exist
    export TEST_BASE=$OUTPUT_TEST_DIR/$DATASET_NAME

    # input directories
    export INPUT_PREP_CHIPS=$PREP_BASE/chips
    export INPUT_PREP_LABELS=$PREP_BASE/labels
    export INPUT_PREP_MULTIMASKS=$PREP_BASE/multimasks

    # output directories
    export OUTPUT_TEST_CHIPS=$TEST_BASE/chips
    export OUTPUT_TEST_LABELS=$TEST_BASE/labels
    export OUTPUT_TEST_MULTIMASKS=$TEST_BASE/multimasks

    # test chip list
    export TEST_CHIP_CSV=$PREP_BASE/test_chips.csv
    
    echo
    echo MOVING IMAGERY TO TEST LOCATIONS: $DATASET_NAME
    echo python $MOVE_EXEC -sd $INPUT_PREP_CHIPS -td $OUTPUT_TEST_CHIPS -csv $TEST_CHIP_CSV $MOVE_FILES
    python $MOVE_EXEC -sd $INPUT_PREP_CHIPS -td $OUTPUT_TEST_CHIPS -csv $TEST_CHIP_CSV $MOVE_FILES
    
    echo
    echo MOVING LABELS TO TEST LOCATIONS: $DATASET_NAME
    echo python $MOVE_EXEC -sd $INPUT_PREP_LABELS -td $OUTPUT_TEST_LABELS -csv $TEST_CHIP_CSV $MOVE_FILES
    python $MOVE_EXEC -sd $INPUT_PREP_LABELS -td $OUTPUT_TEST_LABELS -csv $TEST_CHIP_CSV $MOVE_FILES

    echo
    echo MOVING MULTICHANNEL MASKS TO TEST LOCATIONS: $DATASET_NAME
    echo python $MOVE_EXEC -sd $INPUT_PREP_MULTIMASKS -td $OUTPUT_TEST_MULTIMASKS -csv $TEST_CHIP_CSV $MOVE_FILES
    python $MOVE_EXEC -sd $INPUT_PREP_MULTIMASKS -td $OUTPUT_TEST_MULTIMASKS -csv $TEST_CHIP_CSV $MOVE_FILES

    echo 
    echo

done