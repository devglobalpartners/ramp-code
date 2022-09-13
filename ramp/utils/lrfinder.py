#################################################################
#
# created for ramp project, August 2022
# Author: carolyn.johnston@dev.global
#
#################################################################



from tensorflow.keras.callbacks import LambdaCallback
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt
import numpy as np
import tempfile

# adding logging
import logging
log = logging.getLogger(__name__)
log.addHandler(logging.NullHandler())

import os, sys
from pathlib import Path

RAMP_HOME = os.environ["RAMP_HOME"]

class LearningRateFinder:
    def __init__(self, model, cfg):
        self.model = model
        lrf_cfg = cfg["lrfinder"]
        self.stopFactor = lrf_cfg["stopFactor"] # usually 4
        self.beta = lrf_cfg["beta"]
        self.startLR = lrf_cfg["startLR"]
        self.endLR = lrf_cfg["endLR"]
        self.lrfind_plot_path = str(Path(RAMP_HOME)/lrf_cfg["lrfind_plot_path"])

        # other cfgs
        self.n_epochs = cfg["num_epochs"]
        self.n_train = cfg["runtime"]["n_training"]
        self.batch_size = cfg["batch_size"]
        self.lrs = []
        self.losses = []
        self.lrMult = 1
        self.avgLoss = 0
        self.bestLoss = 1e9
        self.batchNum = 0
        self.weightsFile = None
    
    def reset(self):
        self.lrs = []
        self.losses = []
        self.lrMult = 1
        self.avgLoss = 0
        self.bestLoss = 1e9
        self.batchNum = 0
        self.weightsFile = None

    
    def on_batch_end(self, batch, logs):
        # grab the current learning rate and add log it to the list of
		# learning rates that we've tried
        lr = K.get_value(self.model.optimizer.lr)
        self.lrs.append(lr)
	
    	# grab the loss at the end of this batch, increment the total
		# number of batches processed, compute the average average
		# loss, smooth it, and update the losses list with the
		# smoothed value
        l = logs["loss"]
        self.batchNum += 1
        self.avgLoss = (self.beta * self.avgLoss) + ((1 - self.beta) * l)
        smooth = self.avgLoss / (1 - (self.beta ** self.batchNum))
        self.losses.append(smooth)

		# compute the maximum loss stopping factor value
        stopLoss = self.stopFactor * self.bestLoss

		# check to see whether the loss has grown too large
        if self.batchNum > 1 and smooth > stopLoss:
            # stop training and return from the method
            self.model.stop_training = True
            return

		# check to see if the best loss should be updated
        if self.batchNum == 1 or smooth < self.bestLoss:
            self.bestLoss = smooth

		# increase the learning rate
        lr *= self.lrMult
        K.set_value(self.model.optimizer.lr, lr)


    def find(self, train_batches):
        # reset our class-specific variables
        self.reset()

        epochs = self.n_epochs
        batch_size = self.batch_size
        startLR = self.startLR
        endLR = self.endLR

        # computed parameters
        steps_per_epoch = self.n_train // self.batch_size
        num_batch_updates = epochs * steps_per_epoch
        self.lrMult = (endLR/startLR) ** (1.0/num_batch_updates)
        log.info(f"startLR/endLR: {startLR}/{endLR}")
        log.info(f"num_batch_updates: {num_batch_updates}")
        log.info(f"lrMult: {self.lrMult}")

        # save the initial model weights
        self.weightsFile = tempfile.mkstemp()[1]
        self.model.save_weights(self.weightsFile)

        # grab the *original* learning rate (so we can reset it
		# later), and then set the *starting* learning rate
        origLR = K.get_value(self.model.optimizer.lr)
        K.set_value(self.model.optimizer.lr, startLR)

        # construct the callback that will be called at the end of each
		# batch, enabling us to increase our learning rate as training
		# progresses
        callbacks_list = []
        callbacks_list.append(LambdaCallback(on_batch_end=lambda batch, logs:
			self.on_batch_end(batch, logs)))

        #### DO THE TRAINING ####

        history = self.model.fit(
            train_batches, 
            epochs=epochs, 
            steps_per_epoch=steps_per_epoch, 
            callbacks=callbacks_list)

        # restore the original model weights and learning rate
        # question, do we need this if we aren't training in this script?
		# self.model.load_weights(self.weightsFile)
		# K.set_value(self.model.optimizer.lr, origLR)

    def plot_loss(self, title="", skipBegin=10, skipEnd=1):
        # grab the learning rate and losses values to plot
        lrs = self.lrs[skipBegin:-skipEnd]
        losses = self.losses[skipBegin:-skipEnd]
		
        # plot the learning rate vs. loss
        plt.plot(lrs, losses)
        plt.xscale("log")
        plt.xlabel("Learning Rate (Log Scale)")
        plt.ylabel("Loss")
		# if the title is not empty, add it to the plot
        if title != "":
        	plt.title(title)

    