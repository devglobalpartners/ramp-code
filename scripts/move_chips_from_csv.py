#################################################################
#
# created for ramp project, August 2022
# Author: carolyn.johnston@dev.global
#
#################################################################

 
 
import sys, os, re, argparse, csv, shutil, tqdm
from pathlib import Path

RAMP_HOME = os.environ['RAMP_HOME']

from ramp.utils.file_utils import move_training_data_files

# adding logging
# some options for log levels: INFO, WARNING, DEBUG
import logging
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
log.addHandler(ch)

# Used to check whether directory paths are given in the arguments
def dir_path(string):
    if os.path.isdir(string):
        return string
    else:
        raise NotADirectoryError(string)

def main():

    parser = argparse.ArgumentParser(description='''
    Move or copy image, label, or mask files to new directories based on a list of matching filenames from a csv.

    Examples: move_chips_from_csv.py -sd multimasks -td val-multimasks -csv datasplit_val.csv -mv
    
    Where datasplit_val.csv is a list of chip files, usually made by make_train_val_split_lists.py.
    This call will move files in the multimask directory that match the chip filenames in the csv to the val-multimask directory.
    Leave off the -mv flag if you want them copied, not moved.

    ''',formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-sd', '--src_dir', type=dir_path, required=True, help=r'Path to directory containing TQ geotiffs.')
    parser.add_argument('-td', '--target_dir', type=str, required=True, help=r'Path to target dir for moved chips')
    parser.add_argument('-csv', '--csv', type=str, required=True, help=r'csv file containing chip, label, or mask filenames')
    parser.add_argument('-fp', '--filename_pos', type=int, required=False, default=0, help=r'column number (0-based) of filename in csv file')
    parser.add_argument('-mv', '--move_file', action='store_true', help='use this flag if you want to move files. Default behavior is to copy.')
    args = parser.parse_args()

    # Check that csv exists
    if not Path(args.csv).is_file():
        raise FileNotFoundError(args.csv)
            
    # get list of filenames from csv for files to move
    filenames = []
    with open(args.csv) as csvfile:
        reader = csv.reader(csvfile, delimiter = ',')
        for row in reader:
            filename = row[args.filename_pos]
            filenames.append(filename)

    move_training_data_files(filenames, args.src_dir, args.target_dir, move_file=args.move_file)
    
    return

if __name__=='__main__':
    main()
    


    




