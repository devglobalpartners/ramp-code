
#################################################################
#
# created for ramp project, August 2022
# Author: carolyn.johnston@dev.global
#
#################################################################

 
 
# Convenience script: 
# creates directories of matching multichannel masks
# from a list of datasets of matching chips and labels.

##################################################################
# USER TO SET

# set name of parent directory for all input chip/label dataset directories

export INPUT_PREP_DIR=$RAMP_HOME/ramp-data/PREP

# list of datasets of chips and labels in the RAMP_HOME/ramp-data/PREP directory.
# chips are in INPUT_PREP_DIR/DATASET_NAME/chips
# labels are in INPUT_PREP_DIR/DATASET_NAME/labels
export DATASET_NAMES=(dhaka3)

# width of desired boundaries for buildings, and distance for close contact spacing, in pixel units
# 2 and 4 are standard
export BOUNDARY_WIDTH=2
export CONTACT_SPACING=4
##################################################################

# scripts directory
export SCRIPTS=$RAMP_HOME/ramp-staging/scripts

# create multichannel masks
export MULTIMASK_EXEC=$SCRIPTS/multi_masks_from_polygons.py

for DATASET_NAME in "${DATASET_NAMES[@]}"; do

    echo ==========================================================
    echo
    echo PROCESSING $DATASET_NAME

    export CHIPS_DIR=$INPUT_PREP_DIR/$DATASET_NAME/chips
    export LABELS_DIR=$INPUT_PREP_DIR/$DATASET_NAME/labels
    export MULTIMASK_DIR=$INPUT_PREP_DIR/$DATASET_NAME/multimasks
    echo 
    echo 
    echo Creating multichannel masks for $DATASET_NAME in $MULTIMASK_DIR
    echo python $MULTIMASK_EXEC -in_vecs $LABELS_DIR -in_chips $CHIPS_DIR -out $MULTIMASK_DIR -bwidth $BOUNDARY_WIDTH -csp $CONTACT_SPACING
    python $MULTIMASK_EXEC -in_vecs $LABELS_DIR -in_chips $CHIPS_DIR -out $MULTIMASK_DIR -bwidth $BOUNDARY_WIDTH -csp $CONTACT_SPACING
    echo 
    echo 

done