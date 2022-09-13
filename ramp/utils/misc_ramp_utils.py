
#################################################################
#
# created for ramp project, August 2022
#
#################################################################


import glob, os, re
from pathlib import Path
import csv

# adding logging
import logging
log = logging.getLogger(__name__)
log.addHandler(logging.NullHandler())

def check_create_dir(dirname):
    if not os.path.exists(dirname):
        log.info('Creating directory: {dirname}')
        os.makedirs(dirname, exist_ok=True)

# Used to check whether directory paths are given in the arguments
def dir_path(string):
    if os.path.isdir(string):
        return string
    else:
        raise NotADirectoryError(string)
        
def get_num_files(dirname, pattern = "*.tif"):
    ''' 
    get the number of TIFF files in a directory.
    '''
    return len(glob.glob(os.path.join(dirname, pattern)))

# this is here because a lot of older notebooks import this function but don't use it
def get_experiment_db():
    pass


def log_experiment_to_file(
    the_experiment, 
    the_filename=os.path.join(os.environ['RAMP_HOME'],'ramp-code/ramp_experiments.csv')
    ):
    '''
    log an experiment to an existing or new csv file.
    param: dict the_experiment: key-value store of information to be logged about the experiment. 
    param: str the_filename: file to log the experiment to.  
    '''

    csvpath = Path(the_filename)    
    if csvpath.is_file():
        
        # file exists, open and read
        newrows = []
        fieldnames = []
        with open(str(csvpath),'r') as csvfile:
            reader = csv.DictReader(csvfile)

            ## bugfix: add new fieldnames to the experiment csv
            fieldnames = reader.fieldnames
            for row in reader:
                newrows.append(row)

        expkeys = the_experiment.keys()
        for key in expkeys: 
            if key not in fieldnames:
                fieldnames.append(key)

        newrows.append(the_experiment)

        with open(str(csvpath), 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(newrows)

    else:
        # file does not exist
        with open(str(csvpath), 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=the_experiment.keys())
            writer.writeheader()
            writer.writerows([the_experiment])

    return

def get_chip_number_sn2(the_filename):
    '''
    extract chip number from any data filename in Spacenet 2 format
    '''
    import re
    import os.path

    chipnum_pattern = re.compile(r"(?P<numbers>[0-9]+)$")
    extensions = set('.tiff .tif .json .geojson'.split())
    fbase, ext = os.path.splitext(the_filename)

    if ext not in extensions:
        return None 

    chip_match = chipnum_pattern.search(fbase)
    if chip_match is None:
        log.warning(f"Unhandled filename: {the_filename}")
        return None

    chipnum = int(chip_match.group("numbers"))
    return chipnum