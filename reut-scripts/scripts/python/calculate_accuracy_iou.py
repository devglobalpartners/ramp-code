import os, sys
from pathlib import Path
import rasterio as rio
import numpy as np
import argparse
#################################################################
#
# created for ramp project, August 2022
# Author: carolyn.johnston@dev.global
#
#################################################################

 
 
import tensorflow as tf
from tqdm import tqdm
import geopandas as gpd, pandas as pd

# get rid of geopandas FutureWarning
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# adding logging
# some options for log levels: INFO, WARNING, DEBUG
import logging
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
log.addHandler(ch)

from ramp.utils.eval_utils import get_iou_accuracy_metrics
from ramp.utils.file_utils import get_basename
from ramp.utils.misc_ramp_utils import log_experiment_to_file

def main():

    parser = argparse.ArgumentParser(description=''' 
    Calculate accuracy based on intersection-over-union.
    This script can accept either single truth and detected geojson files,
    or directories of matching truth and detected geojson files. 
    Warning: this routine is very slow for extremely large data sets.

    Example for individual geojson files: 
    
    calculate_accuracy_iou.py -truth truth_labels_filename.json -test predicted_labels_filename.json \\
        -log logfilename.csv -note note_for_log -mid modelname_for_log -dset datasetname_for_log

    Example for directories of matching truth and detected geojson labels:

    calculate_accuracy_iou.py -truth truth_labels_dir -test predicted_labels_dir \\
        -log logfilename.csv -note note_for_log -mid modelname_for_log -dset datasetname_for_log
    
    ''',formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-truth', '--truth_path', type=str, required=True, help=r'Path to a truth label json file or a directory of truth labels.')
    parser.add_argument('-test', '--test_path', type=str, required = True, help='Path to a test label json file or a directory of test labels')
    parser.add_argument('-f_m2', '--filter_area_m2', type=float, required=False, default=None, help='minimum area of buildings to analyze in m^2 (default None)')
    parser.add_argument('-iou', '--iou_threshold', type = float, required = False, default = 0.5, help="IoU threshold for a building detection (default 0.5")
    
    # records for logging results
    parser.add_argument('-log', '--logfile', type=str, required=False, default=None, help="csv file for logging metrics (default None)" )
    parser.add_argument('-note', '--lognote', type=str, required=False, default=None, help='''note to include with records in the logfile''')
    parser.add_argument('-mid', '--model_id', type=str, required=False, default=None, help='''model id to include with records in logfile''')
    parser.add_argument('-dset', '--dataset_id', type=str, required=False, default=None, help='''dataset id to include with records in logfile''')
    
    # for granular, per-file output if a directory is being processed
    parser.add_argument('-csv', '--csvfile', type=str, required=False, default = None, help="csv file for per-file granular output if processing a directory of labels (default None)")

    args = parser.parse_args()

    truth_path = Path(args.truth_path)
    test_path = Path(args.test_path)
    
    if truth_path.is_dir():
        log.info(f"Reading truth label files from directory: {str(truth_path)}")
        assert test_path.is_dir(), f"{test_path} is not a directory" 
        read_from_dir = True
    elif truth_path.is_file():
        log.info(f"Reading truth labels from geojson file: {str(truth_path)}")
        assert test_path.is_file(), f"{test_path} is not a file"
        read_from_dir = False
    else:
        log.error(f"Truth data could not be identified as directory or file: {truth_path}")
        return

    iou_threshold = args.iou_threshold
    
    csvfile = args.csvfile
    write_csvfile = False
    if csvfile is not None:
        write_csvfile = True
        log.info(f"writing building count results to {csvfile}")

    lognote = None
    logfile = args.logfile
    append_to_logfile = False
    if logfile is not None:
        append_to_logfile = True
        log.info(f"Logging aggregated output metrics to {logfile}")

    filter_area_m2 = args.filter_area_m2
    if filter_area_m2:
        log.info(f"Applying building size filtering: minimum {filter_area_m2} sqm")

    n_detections = []
    n_truth = []
    n_truepos = []
    n_falsepos = []
    n_falseneg = []

    #######################################################
    # IF SOURCE DATA CONSISTS OF INDIVIDUAL TILES IN DIRECTORIES
    #######################################################

    if read_from_dir:
        
        # Calculate aggregated accuracy metrics over the set of truth/test file pairs
        truth_label_files = sorted(truth_path.glob('**/*.geojson'))
        test_label_files = sorted(test_path.glob('**/*.geojson'))

        for file in truth_label_files:
            found_match = False
            truth_bname = get_basename(str(file))
            for test_file in test_label_files:
                if get_basename(str(test_file)) == truth_bname:
                    found_match = True
                    truth_df = gpd.read_file(str(file))
                    test_df = gpd.read_file(str(test_file))
                    metrics = get_iou_accuracy_metrics(test_df, truth_df, filter_area_m2, iou_threshold)
                    n_detections.append(metrics['n_detections'])
                    n_truth.append(metrics["n_truth"])
                    n_truepos.append(metrics['true_pos'])
                    n_falsepos.append(metrics['n_detections'] - metrics['true_pos'])
                    n_falseneg.append(metrics['n_truth'] - metrics['true_pos'])
                    break
            if not found_match:
                n_detections.append(None)
                n_truth.append(None)
                n_truepos.append(None)
                n_falsepos.append(None)
                n_falseneg.append(None)
    
        csv_data = {
            "truth_file":[str(file) for file in truth_label_files],
            "n_detections":n_detections,        
            "n_truth":n_truth,
            "n_truepos":n_truepos,
            "n_falsepos":n_falsepos,
            "n_falseneg":n_falseneg
        }

        csv_df = pd.DataFrame(csv_data)

        if write_csvfile:
            csv_df.to_csv(csvfile)

        # drop none columns to get summary statistics
        csv_df = csv_df.dropna()
        all_TP = sum(csv_df["n_truepos"])
        all_truth = sum(csv_df["n_truth"])
        all_detects = sum(csv_df["n_detections"])
        log.info(f"all matches: {all_TP}")
        log.info(f"all detects: {all_detects}")
        log.info(f"all truth: {all_truth}")

        agg_recall = all_TP/all_truth
        agg_precision = all_TP/all_detects
        agg_f1 = 2*all_TP/(all_truth + all_detects)

        log.info(f"Aggregate Precision IoU@p: {agg_precision}")
        log.info(f"Aggregate Recall IoU@p: {agg_recall}")
        log.info(f"Aggregate F1 IoU@p: {agg_f1}")
        
    #######################################################
    # IF SOURCE DATA CONSISTS OF INDIVIDUAL GEOJSON FILES
    #######################################################

    if not read_from_dir:

        truth_df = gpd.read_file(str(truth_path))
        test_df = gpd.read_file(str(test_path))
        metrics = get_iou_accuracy_metrics(test_df, truth_df, filter_area_m2, iou_threshold)

        n_detections = metrics['n_detections']
        n_truth = metrics["n_truth"]
        n_truepos = metrics['true_pos']
        n_falsepos = n_detections - n_truepos
        n_falseneg =  n_truth - n_truepos

        log.info(f"Detections: {n_detections}")
        log.info(f"Truth buildings: {n_truth}")
        log.info(f"True positives: {n_truepos}")

        agg_precision = n_truepos/n_detections
        agg_recall = n_truepos/n_truth
        agg_f1 = 2*n_truepos/(n_truth + n_detections)
        log.info(f"Precision IoU@p: {agg_precision}")
        log.info(f"Recall IoU@p: {agg_recall}")
        log.info(f"F1 IoU@p: {agg_f1}")

    if append_to_logfile:
        explog = {}
        explog["model_id"] = args.model_id
        explog["dataset_id"] = args.dataset_id
        explog["notes"] = args.lognote
        explog["truth_path"] = truth_path
        explog["test_path"] = test_path
        explog["min_bldg_size"] = filter_area_m2
        explog["precision"] = agg_precision
        explog["recall"] = agg_recall
        explog["f1"] = agg_f1
        log_experiment_to_file(explog, logfile)


if __name__=="__main__":
    main()
        

    