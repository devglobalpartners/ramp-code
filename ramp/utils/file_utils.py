#################################################################
#
# created for ramp project, August 2022
# Author: carolyn.johnston@dev.global
#
#################################################################

import os, sys, re, shutil
from random import sample as rsample
from pathlib import Path
import pandas as pd

def get_basename(filepath):
    '''
    recovers the  basename of an image, mask, prediction, or label file 
    in any location in a RAMP training or output dataset.
    i.e., basename.mask.tif -> basename
    basename.tif -> basename
    basename.mask.sdt.tif -> basename
    basename.pred.tif -> basename
    basename.geojson -> basename
    '''
    # the basename ends at the string '.mask' or '.pred'
    # the stem will be basename or basename.mask or basename.mask.sdt 
    filepath = str(filepath)
    the_stem = Path(filepath).stem
    rex = r"(?P<basename>.*)((\.mask)|(\.pred))"
    reg = re.compile(rex)
    m = reg.search(the_stem)
    if m is None:
        # the 'mask' and 'predict' parts didn't match
        return the_stem
    else:
        return m['basename']
 

def get_matching_pairs(d1_iter, d2_iter):
    '''
    Given 2 lists of files, 
    constructs an iterator over matching pairs left-joined on the first list.
    Pairs are matched if get_basename(file1) == get_basename(file2). 
    '''    
    d1_files = [f for f in d1_iter]
    d2_files = [f for f in d2_iter]

    num_d1_files = len(d1_files)

    # sanity check
    assert len(d1_files) == len(d2_files),\
        "Directories contain unequal counts of chip and label files"

    d1_basenames = [get_basename(fn) for fn in d1_files]
    d2_basenames = [get_basename(fn) for fn in d2_files]

    df1 = pd.DataFrame(
        {
            "filename_1":d1_files,
            "basename":[get_basename(fn) for fn in d1_files]
        }
    )
    df2 = pd.DataFrame(
        {
            "filename_2":d2_files,
            "basename":[get_basename(fn) for fn in d2_files]
        }
    )

    joined_df = pd.merge(df1, df2, on="basename", how="left")
    paired_files = [(row["filename_1"], row["filename_2"]) for index, row in joined_df.iterrows()]
    assert len(paired_files) == num_d1_files, \
        "Directories contain mismatched chip and label files"
    return paired_files

def get_SRS_files(sample_size, the_dir, pattern='**/*'):
    '''
    returns a simple random sample (no repeated files) from the directory.
    if sample_size > number files in directory, returns complete list of files. 
    pattern: string to pass to the_dir.glob(pattern) to get the list of files. Default is trivial pattern.  
    '''

    dirpath = Path(the_dir)
    assert dirpath.is_dir(), f"Directory {the_dir} doesn't exist"
    files = dirpath.glob(pattern)

    # select only files, no directories
    filenames = [fn for fn in files if fn.is_file()]

    # return full list
    numfiles = len(filenames)
    if sample_size >= numfiles:
        return files

    file_sample = rsample(filenames, sample_size)
    return file_sample


def move_training_data_files(basename_list, src_dir, target_dir, move_file=False):
    '''
    # todo: change print statements to log statements
    basename_list: list of files to take basenames from, usually from get_SRS_files
    move_file: if true, move the file: if false, copy it
    '''    
    # Create target dirs if they don't exist
    target_path = Path(target_dir)
    target_path.mkdir(parents=True, exist_ok=True)
            
    # get list of basenames from basename_list for files to move
    basenames = []
    for filename in basename_list:
            # get the 'true stem' for files with >1 suffix: 
            # i.e., it handles names like basename.mask.tif correctly
            basename = get_basename(filename)
            basenames.append(basename)

    # copy over files from the source having basenames in 'basenames' list
    src_files_list = Path(src_dir).glob('**/*')
    for sfn in src_files_list:
        sbase = get_basename(sfn)
        if sbase in basenames:
            if move_file:
                #print(f"Move: {sfn} to {target_dir}")
                shutil.move(str(sfn), target_dir)
            else:
                #print(f"Copy: {sfn} to {target_dir}")
                shutil.copy(str(sfn), target_dir)



def get_matching_mask(basename, mask_dir):
    '''
    return full path to 'basename.mask.tif' in mask_dir
    '''
    mdir = Path(mask_dir)
    return mdir / f"{basename}.mask.tif"
    
    


    

