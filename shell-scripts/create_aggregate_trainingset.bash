
#################################################################
#
# created for ramp project, August 2022
# Author: carolyn.johnston@dev.global
#
#################################################################

 
 
## CONVENIENCE SCRIPT: CREATES A COMBINED TRAINING SET FROM A LIST OF EXISTING TRAINING AND TEST SETS.
# This script creates the aggregate directories and populates them with softlinks to items in the source datasets. 
# These are relative softlinks, so they will work in both the host and docker container environments.
# Example of relative softlink: ln -s ../../myfile.txt .
# Example of absolute softlink: ln -s /home/carolyn/myfile.txt .

#################################################################################
# USER SET VALUES
# list of datasets of chips and labels in the RAMP_HOME/ramp-data/TRAIN and RAMP_HOME/ramp-data/TEST directories.
export DATASET_NAMES=(dhaka1 dhaka2 dhaka3)

# name of aggregated dataset
export AGG_DATASET_NAME=dhaka_tq
#################################################################################

# parent directories for all training and testing datasets
export ALL_TRAIN_DIR=$RAMP_HOME/ramp-data/TRAIN
export ALL_TEST_DIR=$RAMP_HOME/ramp-data/TEST

# new training and testing dataset directories to be created
export TRAIN_TARGET=$ALL_TRAIN_DIR/$AGG_DATASET_NAME
export TEST_TARGET=$ALL_TEST_DIR/$AGG_DATASET_NAME

# create training and testing directories for new dataset
mkdir -p $TRAIN_TARGET 
mkdir -p $TEST_TARGET
(cd $TRAIN_TARGET && mkdir -p chips labels valchips val-labels multimasks val-multimasks)
(cd $TEST_TARGET && mkdir -p chips labels multimasks)

# go through input datasets and softlink every file to the new location
for DATASET_NAME in "${DATASET_NAMES[@]}"; do

    echo 
    echo PROCESSING $DATASET_NAME
    echo 

    # We have to make RELATIVE softlinks!
    export TRAIN_SRC=../../$DATASET_NAME
    export TEST_SRC=../../$DATASET_NAME
    echo $TRAIN_SRC

    cd $TRAIN_TARGET
    pwd

    (cd chips && ln -s $TRAIN_SRC/chips/* .)
    (cd labels && ln -s $TRAIN_SRC/labels/* .)
    (cd valchips && ln -s $TRAIN_SRC/valchips/* .)
    (cd val-labels && ln -s $TRAIN_SRC/val-labels/* .)
    (cd multimasks && ln -s $TRAIN_SRC/multimasks/* .)
    (cd val-multimasks && ln -s $TRAIN_SRC/val-multimasks/* .)
    
    cd $TEST_TARGET
    pwd 
    
    (cd chips && ln -s $TEST_SRC/chips/* .)
    (cd labels && ln -s $TEST_SRC/labels/* .)
    (cd multimasks && ln -s $TEST_SRC/multimasks/* .)

    echo DONE
    echo 
    echo

done