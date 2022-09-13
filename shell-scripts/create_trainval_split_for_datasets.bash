#################################################################
#
# created for ramp project, August 2022
# Author: carolyn.johnston@dev.global
#
#################################################################

 
 
# Convenience script to define a train/validation split of a dataset,
# and move these sets to their final locations,
# for each dataset in a list.

# Details: For each dataset of matching geotiffs and geojsons in a list of datasets, 
# This script randomly splits them into training and validation sets. 
# THE TEST SET SHOULD ALREADY HAVE BEEN CREATED and all test chips and labels moved from this dataset.
# The 'create_test_split.bash' script can be used for this purpose.

# for each dataset in the DATASET_NAMES list,
# training and validation chips are randomly subsampled from a dataset in the PREP_BASE directory.
# all validation data (both chips and labels) is moved to subdirectories in TRAIN_BASE named valchips/val-labels.
# the training data is moved to subdirectories in TRAIN_BASE named chips/labels.

##########################################################
# USER TO SET 

# parent directory for all input chip/label datasets to be split into training/validation datasets
# chips are in INPUT_PREP_DIR/DATASET_NAME/chips
# labels are in INPUT_PREP_DIR/DATASET_NAME/labels
export INPUT_PREP_DIR=$RAMP_HOME/ramp-data/PREP

# parent directory for all output chip/label training and validation datasets
# training chips are output to: OUTPUT_TRAIN_DIR/DATASET_NAME/chips
# training labels are output to: OUTPUT_TRAIN_DIR/DATASET_NAME/labels
# validation chips are output to: OUTPUT_TRAIN_DIR/DATASET_NAME/valchips
# validation labels are output to: OUTPUT_TRAIN_DIR/DATASET_NAME/val-labels
export OUTPUT_TRAIN_DIR=$RAMP_HOME/ramp-data/TRAIN

# list of directories of chips and labels datasets in the INPUT_PREP_DIR directory.
export DATASET_NAMES=(coxbzr2)

# rootname of files containing lists of training and validation chips.
export FILE_ROOTNAME=tv_split

# comment out this line if you want the data files to be copied, not moved, from the source directory.
export MOVE_FILES=-mv

# training and validation fractions should sum to 1
export TRAIN_FRAC=0.85
export VAL_FRAC=0.15
##########################################################

# scripts to execute
export SCRIPTS=$RAMP_HOME/ramp-staging/scripts

# do the random sampling for the train_val split and create lists of files
export SPLIT_EXEC=$SCRIPTS/make_train_val_split_lists.py

# move the chips and labels based on a list of chips in a csv file. 
export MOVE_EXEC=$SCRIPTS/move_chips_from_csv.py

echo CREATING DATASETS FOR TRAINING AND VALIDATION 
echo FRACTION OF DATA TO USE FOR TRAINING DATA: $TRAIN_FRAC
echo FRACTION OF DATA TO USE FOR VALIDATION DATA: $VAL_FRAC
echo 
echo 

for DATASET_NAME in "${DATASET_NAMES[@]}"; do

    echo ==========================================================
    echo
    echo
    echo PROCESSING $DATASET_NAME

    export PREP_BASE=$INPUT_PREP_DIR/$DATASET_NAME
    export TRAIN_BASE=$OUTPUT_TRAIN_DIR/$DATASET_NAME

    # source directories for chips and labels
    export PREP_CHIPS=$PREP_BASE/chips
    export PREP_LABELS=$PREP_BASE/labels

    # target directories for chips and labels
    export TRAIN_CHIP_DIR=$TRAIN_BASE/chips
    export TRAIN_LABEL_DIR=$TRAIN_BASE/labels
    export VAL_CHIP_DIR=$TRAIN_BASE/valchips
    export VAL_LABEL_DIR=$TRAIN_BASE/val-labels

    # source directory for masks
    export PREP_MMASKS=$PREP_BASE/multimasks

    # target directories for masks
    export TRAIN_MMASK_DIR=$TRAIN_BASE/multimasks
    export VAL_MMASK_DIR=$TRAIN_BASE/val-multimasks

    echo
    echo
    echo MAKING TRAINING AND VALIDATION CHIP LISTS
    echo python $SPLIT_EXEC -src $PREP_CHIPS -pfx $FILE_ROOTNAME -trn $TRAIN_FRAC -val $VAL_FRAC
    python $SPLIT_EXEC -src $PREP_CHIPS -pfx $FILE_ROOTNAME -trn $TRAIN_FRAC -val $VAL_FRAC

    echo 
    echo
    echo MOVING IMAGERY TO TRAINING LOCATIONS
    echo python $MOVE_EXEC -sd $PREP_CHIPS -td $TRAIN_CHIP_DIR -csv ${FILE_ROOTNAME}_train.csv $MOVE_FILES
    python $MOVE_EXEC -sd $PREP_CHIPS -td $TRAIN_CHIP_DIR -csv ${FILE_ROOTNAME}_train.csv $MOVE_FILES  
    echo python $MOVE_EXEC -sd $PREP_CHIPS -td $VAL_CHIP_DIR -csv ${FILE_ROOTNAME}_val.csv $MOVE_FILES
    python $MOVE_EXEC -sd $PREP_CHIPS -td $VAL_CHIP_DIR -csv ${FILE_ROOTNAME}_val.csv $MOVE_FILES

    echo
    echo
    echo MOVING LABELS TO TRAINING LOCATIONS
    echo python $MOVE_EXEC -sd $PREP_LABELS -td $TRAIN_LABEL_DIR -csv ${FILE_ROOTNAME}_train.csv $MOVE_FILES
    python $MOVE_EXEC -sd $PREP_LABELS -td $TRAIN_LABEL_DIR -csv ${FILE_ROOTNAME}_train.csv $MOVE_FILES
    echo python $MOVE_EXEC -sd $PREP_LABELS -td $VAL_LABEL_DIR -csv ${FILE_ROOTNAME}_val.csv $MOVE_FILES
    python $MOVE_EXEC -sd $PREP_LABELS -td $VAL_LABEL_DIR -csv ${FILE_ROOTNAME}_val.csv $MOVE_FILES

    echo
    echo
    echo MOVING MULTICHANNEL MASKS TO TRAINING LOCATIONS
    echo python $MOVE_EXEC -sd $PREP_MMASKS -td $TRAIN_MMASK_DIR -csv ${FILE_ROOTNAME}_train.csv $MOVE_FILES
    python $MOVE_EXEC -sd $PREP_MMASKS -td $TRAIN_MMASK_DIR -csv ${FILE_ROOTNAME}_train.csv $MOVE_FILES
    echo python $MOVE_EXEC -sd $PREP_MMASKS -td $VAL_MMASK_DIR -csv ${FILE_ROOTNAME}_val.csv $MOVE_FILES
    python $MOVE_EXEC -sd $PREP_MMASKS -td $VAL_MMASK_DIR -csv ${FILE_ROOTNAME}_val.csv $MOVE_FILES

    echo
    echo
    echo COPYING TRAINING DATA LISTS TO TRAINING DIRECTORIES
    echo cp ${FILE_ROOTNAME}_train.csv ${FILE_ROOTNAME}_val.csv $TRAIN_BASE
    cp ${FILE_ROOTNAME}_train.csv ${FILE_ROOTNAME}_val.csv $TRAIN_BASE
    echo DONE
done