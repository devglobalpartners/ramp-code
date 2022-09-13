#################################################################
#
# created for ramp project, August 2022
# Author: carolyn.johnston@dev.global
#
#################################################################

 
 
import sys, os, re, argparse, csv, shutil, tqdm, random
from pathlib import Path

RAMP_HOME = os.environ['RAMP_HOME']

# adding logging
# some options for log levels: INFO, WARNING, DEBUG
import logging
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
log.addHandler(ch)

def main():

    parser = argparse.ArgumentParser(description=''' 
    Create randomized lists of files for training and validation datasets, and save them to files.
    Test datasets should already have been set aside.

    Example: make_train_val_split_lists.py -src chips -trn 0.85 -val 0.15
    ''',formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-src', '--src_dir', type=str, required=True, help=r'Path to directory containing TQ geotiffs.')
    parser.add_argument('-pfx', '--list_prefix', type=str, required=False, default="datasplit", help=r'prefix for train, validation and test file lists')
    parser.add_argument('-trn', '--train_fraction', type=float, required=True, help='fraction of files to assign to training set')
    parser.add_argument('-val', '--val_fraction', type=float, required=False, default=0.0, help='fraction of files to assign to validation set')
    args = parser.parse_args()

    trn = args.train_fraction
    val = args.val_fraction
    
    # force fractions to sum to 1
    normalizer = trn + val
    if normalizer == 0.0:
        raise ValueError("train and validation fractions must sum to 1")
    trn = trn/normalizer
    val = val/normalizer
    
    # check that source directory exists and is readable
    src_dir = args.src_dir
    if not Path(src_dir).is_dir():
        raise ValueError(f"source directory {src_dir} is not readable")

    # construct filenames for file lists
    file_rootname = args.list_prefix
    trn_csv = file_rootname + '_train.csv'
    val_csv = file_rootname + '_val.csv'

    # construct list of all filenames in src dir, shuffle in place
    files = list(Path(src_dir).glob('**/*'))
    random.shuffle(files)      
    numfiles = len(files)
    
    # get numbers of files
    numval = int(val*numfiles)
    numtrain = numfiles - numval

    trainlist = files[:numtrain]
    vallist = files[numtrain:]


    with open(trn_csv, 'w') as trnfp:
        log.info(f"Writing {trn_csv}")
        trnfp.write('\n'.join([str(item) for item in trainlist]))   
    
    if numval !=0:
        with open(val_csv, 'w') as valfp:
            log.info(f"Writing {val_csv}")
            valfp.write('\n'.join([str(item) for item in vallist]))

if __name__=="__main__":
    main()



