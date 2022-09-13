#################################################################
#
# created for ramp project, August 2022
# Author: carolyn.johnston@dev.global
#
#################################################################


import sys, os, re, csv
from pathlib import Path

# Filter functions for csv

def filter_accepted_labels(csvfile, fn_pos=0, accepted_pos=-1, filter_rejected=False):
    '''
    create a list of accepted label filenames from a chip_review.json file created by chippy checker
    existing_csv: list of csv rows returned by csv.Reader
    fn_pos: position of the filename column in csvfile (default 0)
    filter_rejected: set to 'True' if you want to create a list of rejected labels
    '''
    filtered_labels = []
    with open(csvfile, 'r') as fileh:
        reader = csv.reader(csvfile, delimiter = ',')
        for row in reader:
            filename = Path(row[fn_pos])
            accepted = bool(row[accepted_pos])

        condition = accepted if (not filter_rejected) else (not accepted)
        if condition:
            filtered_labels.append(str(filename))
    return filtered_labels

def get_lonlat_from_filename(i_filename):
    '''
    get lon lat (or easting northing) of the upper left hand corner (I think) from 
    chip filenames created by the solaris tiler
    example: 'sn_paris_121_2.212736_49.020317.geojson' returns (2.212736, 49.020317)
    
    i_filename: the filename
    returns: (lon, lat) or None
    '''
    try:
        basename = Path(i_filename).stem
        re_str = r".*_(?P<lon>\-?[0-9]+\.?[0-9]+)_(?P<lat>\-?[0-9]+\.?[0-9]+)$" 
        rex = re.compile(re_str)
        match = rex.search(basename)
        lat = float(match.group('lat'))
        lon = float(match.group('lon'))
        return (lon, lat)
    except:
        return None

