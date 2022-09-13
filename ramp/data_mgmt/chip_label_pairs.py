#################################################################
#
# created for ramp project, August 2022
# Author: carolyn.johnston@dev.global
#
#################################################################

import os, re

def construct_mask_filepath(mask_dir, path_to_chip):
    '''
    Constructs a standard mask file path given the mask directory and the chip filename.
    Standard masks have the name: (chip_basename).mask.tif.
    This means they will alphabetically sort the same way as the image chips did.
    '''

    chip_filename = os.path.split(path_to_chip)[1]
    chip_basename = os.path.splitext(chip_filename)[0]
    mask_file = chip_basename + '.mask.tif'
    return os.path.join(mask_dir, mask_file)

def get_tq_chip_label_pairs(chip_dir, label_dir):
    '''
    Get a list of pairs of matching chips and labels from directories.
    Checks to be sure all chips matching delivered labels are present.

    Matched files are named (e.g.) 84930efd.tif and 84930efd.geojson.

    :param directory chip_dir: directory containing image chips
    :param directory label_dir: directory containing geojson vectors

    Returns: list of tuples of chip and label filename pairs.
    '''

    # construct list of matching image chips from json labels

    # get sorted geojson file names
    gj_files = sorted([gj_file for gj_file in os.listdir(label_dir) if gj_file.endswith('.geojson')])
    
    # get matching chip file names -- for TQ deliveries these have the same base filename 
    basenames = [os.path.splitext(gj_file)[0] for gj_file in gj_files]
    # print(basenames[0])
    chip_files = [basename + '.tif' for basename in basenames]

    # get full pathnames to json and chip files
    json_paths = [os.path.join(label_dir, gj_file) for gj_file in gj_files]
    chip_paths = [os.path.join(chip_dir, chip_file) for chip_file in chip_files]

    # check to be sure the matching image chips are present
    for chip_path in chip_paths:
        if not os.path.exists(chip_path):
            raise FileNotFoundError(f'Missing: required image chip file {chip_path}')
    
    return list(zip(chip_paths, json_paths))

