#################################################################
#
# created for ramp project, August 2022
# Author: carolyn.johnston@dev.global
#
#################################################################



import albumentations as A
import cv2
import re

# adding logging
import logging
log = logging.getLogger(__name__)
log.addHandler(logging.NullHandler())

def get_augmentation_fn(cfg):
    aug_name_list = cfg["augmentation"]["aug_list"]
    aug_parms_list = cfg["augmentation"]["aug_parms"]

    # construct augmentation functions
    aug_fn_list = []
    for augment_name, aug_parms in zip(aug_name_list, aug_parms_list):
        aug_fn = getattr(A, augment_name)

        # some of these values are cv2 flag strings that have to be converted
        flag_parms = { k: getattr(cv2,v) for (k,v) in aug_parms.items() 
            if (isinstance(v, str) and hasattr(cv2, v))}
        aug_parms = {k: v for (k,v) in aug_parms.items() if not isinstance(v, str)}
        aug_parms.update(flag_parms)
        aug_fn_list.append(aug_fn(**aug_parms))
    return A.Compose(aug_fn_list)

        
