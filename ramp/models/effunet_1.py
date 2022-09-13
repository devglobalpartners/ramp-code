#################################################################
#
# created for ramp project, August 2022
# Author: carolyn.johnston@dev.global
#
#################################################################


import segmentation_models as sm


BACKBONE = 'efficientnetb0'
CLASSES = ['building']

# This call appears to retrieve the names of some of 
# the layers of the backbone model. 
# preprocess_input = sm.get_preprocessing(BACKBONE)

def get_effunet(backbone=BACKBONE,
                classes=CLASSES, 
                encoder_freeze=False):

    # define network parameters

    n_classes = len(classes)
    # n_classes = 1 if len(classes) == 1 else (len(classes) + 1)  # case for binary and multiclass segmentation
    activation = 'sigmoid' if n_classes == 1 else 'softmax'

    #create model
    model = sm.Unet(backbone, classes=n_classes, activation=activation, encoder_freeze=encoder_freeze)
    return model


