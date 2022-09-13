#################################################################
#
# created for ramp project, August 2022
#
#################################################################



import numpy as np

def get_best_model_value_and_epoch(model_history):
    '''
    argument: the history object returned by model.fit.
    '''
    val_acc = model_history.history["val_sparse_categorical_accuracy"]
    best_epoch = np.argmax(val_acc)
    best_value = val_acc[best_epoch]
    return best_epoch, best_value