#################################################################
#
# created for ramp project, August 2022
# Author: carolyn.johnston@dev.global
#
#################################################################


import numpy as np
import geopandas as gpd
import pandas as pd
from solaris.eval.iou import calculate_iou
from pathlib import Path
import os, sys

# adding logging
import logging
log = logging.getLogger(__name__)
log.addHandler(logging.NullHandler())

def get_iou_df(test_df, truth_df,iou_thresh=0.5, remove_matching_element=True):
    '''
    returns dataframe of test polygons with a column 'match_id' added
    its value for each test polygon is the id of the best match in the truth dataset,
    or -1 if no match was found
    '''
    test_result_list = []

    # we need to use a copy of the truth data, as 
    # matches will be removed during processing.
    copy_truth_df = truth_df.copy(deep=True)

    # insert columns 'match_id' and 'best_iou_score' into the test dataframe
    test_df = test_df.assign(match_id=pd.array(np.full(len(test_df), -1), dtype="int32"))
    test_df = test_df.assign(best_iou_score=pd.array(np.full(len(test_df), np.nan), dtype="float32"))

    test_df = test_df.reset_index()  # make sure indexes pair with number of rows
    for index, row in test_df.iterrows():
        test_poly = row['geometry']

        # get an iou dataframe containing all the rows in the truth data 
        # with polygons that overlap the test polygon
        # this may be empty but iou_df should never be None
        iou_df = calculate_iou(test_poly, copy_truth_df)
        if len(iou_df) == 0:
            row['match_id'] = -1
            row['best_iou_score'] = np.nan
            test_result_list.append(row)
            continue
        max_iou_row = iou_df.loc[iou_df['iou_score'].idxmax(axis=0, skipna=True)]
        best_iou_score = max_iou_row['iou_score']
        if max_iou_row is None or best_iou_score < iou_thresh:
            row['match_id'] = -1
            row['best_iou_score'] = np.nan
            test_result_list.append(row)
            continue
        row['match_id'] = int(max_iou_row.name)
        row['best_iou_score'] = best_iou_score
        if remove_matching_element:
            copy_truth_df.drop(max_iou_row.name, axis=0, inplace=True)
        test_result_list.append(row)

    return  gpd.GeoDataFrame(test_result_list)

def get_iou_accuracy_metrics(test_data, truth_data, size_threshold=None, iou_threshold=0.5):
    '''
    returns dictionary containing accuracy metrics

    test_data: geopandas data frame containing proposal data
    truth_data: geopandas data frame containing truth data
    -- > Both the truth data and test data frames are assumed to have an 'id' field
    
    threshold (float, 0<threshold<1): above which value of IoU a detection is considered to have occurred
    '''

    f_truth_data = truth_data
    f_test_data = test_data

    # apply filtering
    if size_threshold:
        if "area_m2" in truth_data.keys() and "area_m2" in test_data.keys():
            f_truth_data = truth_data[truth_data["area_m2"] >= size_threshold]
            f_test_data = test_data[test_data["area_m2"] >= size_threshold]
        else:
            log.warning("No area_m2 attribute on buildings: cannot apply building size filtering")
    n_truth = len(f_truth_data)
    n_test = len(f_test_data)
    if n_test == 0 or n_truth==0:
        metrics = {
            'n_detections': n_test, 
            'n_truth':n_truth, 
            'true_pos':0,
            'false_pos':n_test,
            'false_neg':n_truth,
            'recall': 0 if n_truth != 0 else None,
            'precision': 0 if n_test != 0 else None,
            'F1': None
        }
        return metrics

    iou_df = get_iou_df(f_test_data, f_truth_data, iou_thresh=iou_threshold)
    assert len(f_truth_data) == n_truth
    true_positives = iou_df[iou_df['match_id'] != -1]
    n_matched = len(true_positives)
    recall = n_matched/n_truth
    precision = n_matched/n_test
    F1 = 2*recall*precision/(recall+precision) if recall+precision != 0 else 0
    metrics = {
        'n_detections': n_test, 
        'n_truth':n_truth, 
        'true_pos':n_matched,
        'false_pos':n_test - n_matched,
        'false_neg':n_truth - n_matched,
        'recall':recall,
        'precision':precision,
        'F1':F1
    }
    return metrics
