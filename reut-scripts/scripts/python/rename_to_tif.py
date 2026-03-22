#################################################################
#
# created for ramp project, August 2025
# Author: Reut Keller Azulay, kellerr@who.int
#
#################################################################

 
import argparse
import os
from tqdm import tqdm

def main():

    parser = argparse.ArgumentParser(description='''
    Renames all files in the specified directory to include a .tif suffix, as
    required by the RAMP code
    
    Example: rename_to_tif.py -dir path/to/directory
    ''', 
    formatter_class=argparse.RawTextHelpFormatter)

    # required
    parser.add_argument('-dir', '--directory', type=str,
                        required=True, help=r'Path to input images.')
    
    args = parser.parse_args()
    directory = args.directory
    
    rename_to_tif(directory)
    
    
    

def rename_to_tif(directory:str , #path to directory with .TIF or anything that is not .tif located
                 ):
    
    for filename in os.listdir(directory):
        old_path = os.path.join(directory, filename)
        
        # skip if it's already .tif
        if filename.endswith(".tif"):
            print(f"file name:{filename} ends with .tif , skipping")
            continue
        
        name, ext = os.path.splitext(filename)
        # print(name,ext)
        
        new_filename = name + ".tif"
        new_path = os.path.join(directory, new_filename)
        # print(new_path)
        
        print(f"Renaming: {old_path} -> {new_path}")
        os.rename(old_path, new_path)
    
if __name__=="__main__":
    main()