# Using the ramp training configuration file
#### Carolyn Johnston, June 28, 2022

## TODO: Update.

## Introduction

All model training runs are run from the script ‘train_ramp.py’ in the ramp-code/scripts directory. Since there are many configurable elements in every training run, ramp uses a json configuration file to configure a training run. This configuration file is the only command-line switch that train_ramp.py needs to use. The sections below first discuss high-level things to know about the configuration file, and then go into detail on the options for each configuration element. 

## Important Note: directory names in the configuration file are defined relative to RAMP_HOME

The configuration file includes paths to many directories containing data elements such as training, validation, and test data, and model definitions. It is important to realize that these must all be defined relative to the RAMP_HOME environment variable, which must exist in any environment running ramp.

All your data, code, models, and logs must be in subdirectories of RAMP_HOME. RAMP_HOME is defined as '/tf' in the ramp docker image. On your host machine it will typically be your home directory. The ramp-staging repository will be located in RAMP_HOME.

To check whether RAMP_HOME is defined in your own shell on Linux, type at the command line:

```code
echo $RAMP_HOME
```

The dollar sign in front of 'RAMP_HOME' causes the base shell to output the value of the RAMP_HOME variable. You should see an absolute path to a directory, such as '/tf' or '/home/carolyn'. If you see nothing, then RAMP_HOME is not defined.

To define RAMP_HOME in your own shell on Linux, type at the command line:

```code
export RAMP_HOME=/my_ramp_home
```

The following is an example of a configuration file block defining training and validation datasets. Their paths are assumed to start from RAMP_HOME. So if RAMP_HOME is defined as '/home/user' on your machine, train_ramp.py will look for this directory at '/home/user/ramp-data/TRAIN/Shanghai-Paris-Oman/chips'.  

```code
    "datasets":{
        "train_img_dir":"ramp-data/TRAIN/Shanghai-Paris-Oman/chips",
        "train_mask_dir":"ramp-data/TRAIN/Shanghai-Paris-Oman/bin-masks",
        "val_img_dir":"ramp-data/TRAIN/Shanghai-Paris-Oman/valchips",
        "val_mask_dir":"ramp-data/TRAIN/Shanghai-Paris-Oman/val-binmasks"
    }
```

## Recommendations for setting up your ramp environment

When setting your environment up to run ramp, we recommend that you check out the ramp codebase directly into your RAMP_HOME directory. 

Furthermore, we recommend that you set up all your training data in a subdirectory of RAMP_HOME named 'ramp-data'.  

In addition to lots of input training data, there is a lot of output data associated with every training run. Ramp training runs produce model checkpoints (which ensure that your best models are saved for later use), tensorboard logs (which allow you to monitor the progress of training), and diagnostic plots, all of which can run into GB of supporting data.   

We recommend organizing subdirectories containing training run outputs according to the training dataset you are using for the training run. For example, if you are working with a Bangladesh training data set and a Sierra Leone training data set, you should place your training data samples into separate directories named '$RAMP_HOME/ramp-data/Bangladesh' and '$RAMP_HOME/ramp-data/SierraLeone'. This is discussed in more detail in the 'datasets' section below. 

With this setup, you know that any two training run outputs in the same place are associated with the same training data, and are therefore comparable to each other. 

In order to help with managing the data associated with training, the script 'train_ramp.py' is set up so that a timestamp is associated with every experiment; data produced by the training run is saved in subdirectories that are named using the timestamp. 

For example, if your configuration file states that the tensorboard logging directory is 'ramp-data/Bangladesh/logs', then the tensorboard logs from a training run with the timestamp '20220618-174410' will be written to the directory '$RAMP_HOME/ramp-data/Bangladesh/logs/log-20220618-174410'.

If your configuration file states that the model checkpoint directory is 'ramp-data/ Bangladesh/model-checkpts', then the model checkpoint files from a training run with the timestamp '20220618-174410' will be written to the directory '$RAMP_HOME/ramp-data/Bangladesh/model-checkpts/20220618-174410'.

## Training configuration top-level blocks

#### Experiment name

#### logging

#### datasets

#### num_classes

#### num_epochs

#### batch_size

#### input_img_shape

#### output_img_shape

#### loss

#### metrics

#### optimizer

#### model

#### saved_model

#### augmentation

#### early_stopping

#### cyclic_learning_scheduler

#### tensorboard

#### prediction_logging

#### model_checkpts

#### random_seed
