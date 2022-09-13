#################################################################
#
# created for ramp project, August 2022
# Author: carolyn.johnston@dev.global
#
#################################################################

 
 
# Given a single chip dataset and a single trained model, 
# generate a predicted set of building labels for the full AOI.

##############################################################

# USER TO SET

# directory of input image chips
export CHIPS_DIR=/tf/ramp-data/ramp_ghana_workshops/ghana_test_data/chips

# path to single model for generating predictions
export MODEL_PATH=/tf/ramp-data/ramp_ghana_workshops/ghana_best_model/model_20220826-195852_042_0.972.tf

# directory of predicted masks
export MASKS_DIR=/tf/ramp-data/ramp_ghana_workshops/ghana_test_data/predicted_masks

# the output geojson file of predicted labels
export PREDICTION_JSON=/tf/ramp-data/ramp_ghana_workshops/ghana_test_data/predicted_ghana.json

# width of building boundaries in pixels when model was trained
export BDRY_WIDTH_PIX=2

# temporary directory to write reprojected masks to 
# these will be removed at the end of processing
export TEMP_DIR=/tf/ramp-data/ramp_ghana_workshops/ghana_test_data/temp

# END USER-SET VARIABLES
##############################################################

# scripts to execute
export SCRIPTS=$RAMP_HOME/ramp-staging/scripts
export PRED_MASKS_EXEC=$SCRIPTS/get_model_predictions.py
export MASKS_TO_LABELS_EXEC=$SCRIPTS/get_labels_from_masks.py

# note: the fusion process creates a temporary directory containing predicted masks, reprojected to WGS84 lat long.
# this directory is cleaned up at the end of this script.
export TMP_REPROJECTED_MASK_PATH=$TEMP_DIR/predicted_masks_4326

echo GENERATING MASK PREDICTIONS FROM MODEL
echo 
echo python $PRED_MASKS_EXEC -mod $MODEL_PATH -idir $CHIPS_DIR -odir $MASKS_DIR
python $PRED_MASKS_EXEC -mod $MODEL_PATH -idir $CHIPS_DIR -odir $MASKS_DIR
echo 
echo

echo CREATING FUSED OUTPUT BUILDING POLYGONS
echo 
echo python $MASKS_TO_LABELS_EXEC -in $MASKS_DIR -out $PREDICTION_JSON -tmpdir $TMP_REPROJECTED_MASK_PATH -bwpix $BDRY_WIDTH_PIX
python $MASKS_TO_LABELS_EXEC -in $MASKS_DIR -out $PREDICTION_JSON -tmpdir $TMP_REPROJECTED_MASK_PATH -bwpix $BDRY_WIDTH_PIX

echo CLEANING TEMP DIRECTORY
rm -rf $TMP_REPROJECTED_MASK_PATH