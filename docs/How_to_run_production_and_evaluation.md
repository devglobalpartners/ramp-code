# How to run production and evaluation for the ramp project

## Introduction

After you have trained a ramp model for detecting buildings in satellite imagery, you will want to use the resulting model to output building polygons for use in your projects. You will also want to assess the accuracy of the model for extracting building polygons. This document will detail the production and accuracy assessment processes.

Producing output polygons using a trained ramp model has the following high-level steps. Given a satellite image to be used for production, the user must:

- Cut the satellite image into 'tiles' that are 256x256, the same size as the individual training chips
- Use the trained model to calculate matching 4-channel mask images from the tiles
- Post-process the 4-channel mask images, which form a covering of the original satellite image, into a single set of building polygons covering the satellite image.

if truth building polygons are available for the satellite image, a final step is to compare the detected building polygons with the truth building polygons, and assess the accuracy of the detection process. It will typically not be the case that there is truth data for the entire satellite image, but a small portion or portions of it should have truth data available for validating the results.  


### The RAMP_HOME environment variable

In each environment running ramp, an environment variable called RAMP_HOME needs to be defined. RAMP_HOME is the parent directory for all of your data and code directories in any environment where ramp is run. For example, in the ramp Docker container, RAMP_HOME is defined to be '/tf' because this is the root directory for the Tensorflow Docker container that the ramp Docker container derives from.

All of my data for the ramp project is stored in a subdirectory of RAMP_HOME named 'ramp-data', so that in the docker container, it's all in ''/tf/ramp-data'. 

The ramp codebase is stored in a subdirectory of RAMP_HOME named 'ramp-code'.

The rest of this document assumes that you have the RAMP_HOME environment variable set, and that it contains the ramp-code directory, and a ramp-data directory. 

# Running production using a trained model and new image data

In order to run production in ramp, I will need these two components:

1. A trained ramp model, saved in .tf format
2. New satellite imagery, on which the model will be run to produce new multichannel masks which will then be post-processed to building labels.

The ramp model needs to have been trained on datasets that are similar to the production data; the details of this have been discussed at length elsewhere. 

## TL;DR

The production process described here can be run for a single dataset with an edited version of the shell script *ramp-code/shell-scripts/run_production_on_single_dataset.bash.*

It can also be run for multiple datasets at once, with an edited version of the shell script *ramp-code/shell-scripts/run_production_on_datasets.bash.*

The latter will generate a fused prediction geojson dataset for each of the AOIs that you specify.


### Step 1: Set up your production data directories.

Your production directory structure will need to keep track of the model that you used to run production, as well as the dataset that you ran it on. 

Recall that the RAMP_HOME directory contains two subdirectories in any environment: *ramp-code*, where the ramp github codebase is located, and *ramp-data*, where all the data associated with training and production are located. 

You will need to create a new directory named 'PROD' under *ramp-data*, so your tree should now have these subdirectories:

```
-ramp-data
	--PREP
	--PROD
	--TRAIN
	--TEST
```

Under the PROD directory, create a subdirectory that will correspond to the unique imagery dataset that you will be using for production. Suppose that you are producing building models for an image over Ghana; then you might name this directory *ghana* (note that the name itself doesn't matter; only that it is unique. If you also use a different image over Ghana for production, you could name it *ghana2*).

Now, place the image or images you will be using for production in a subdirectory of *ghana* called *images*. In the next step, you will cut these images up into image chips of size 256x256; these chips will go into a subdirectory named *chips*. 

If you have imagery that has already been tiled into 256x256 image chips, then just create the *chips* directory; you won't need the *images* directory. You can skip Step 2 in that case. 

Your PROD directory will now look something like this:

```
-ramp-data
	--PROD
		--ghana
			--images
				--big_image_of_ghana.tif
			--chips
```

The chips directory will be empty unless your imagery came already cut into image chips, in which case your directory tree should look like this:

```
-ramp-data
	--PROD
		--ghana
			--chips
				- chip1.tif
				- chip2.tif
				- ...
				- chipN.tif
```

### Step 2: Tile the imagery into image chips of size 256x256.

If your images (and accompanying labels if any) are already in the form of image chips, you can skip this step.

The *ramp-code* directory contains a python script, *tile_datasets.py*, which can be used to chop an image into image chips. If there is an accompanying building polygon truth data set (as in the case where you will be evaluating the model's building extraction accuracy), the same script will tile both the image and polygon file into image chips and matching labels.

#### Tiling an image

To tile an image without an accompanying polygon truth file, use the following program call, starting from the RAMP_HOME directory (note the backslashes indicate line continuations in the Linux Bash shell -- they are there only to make this command line easier to read):

```
python ramp-code/scripts/tile_datasets.py \
	-img ramp-data/PROD/ghana/images/big_image_of_ghana.tif \
	-out ramp-data/PROD/ghana/tiled \
	-pfx ghana_chip \
	-trec ramp-data/PROD/ghana/ghana_chips.geojson \
	-ndt 0.4 \
	-oht 256 \
	-owd 256
```
- *img:* the filename of the image to be tiled.
- *out*: the path to the directory where the image chip directory will be created. In this case, image chips will be written to the directory *ramp-data/PROD/ghana/tiled/chips.*
- *pfx*: the prefix of all image chip files to be written. For example, if the prefix is 'ghana_chip' as shown above, the chip files will be named *ghana_chip1.tif*, *ghana_chip2.tif*, etc.
- *trec*: the path to a geojson file containing the boundaries and attributes (including the image chip name) associated with each image chip. Creating this file is optional, but it is extremely useful. 
- *ndt*: No data threshold. If an image chip contains more than this fraction of invalid pixels, it will not be written (this will frequently happen if there is a sliver of valid data on the edge of the larger image).
- *oht, owd*: Height and width, in pixels, of the chips to create.
 
#### Tiling an image with its truth label dataset

To tile an image and an accompanying truth dataset, add a -vecs path to the above program call, pointing to the geojson file containing the truth data:

```
python ramp-code/scripts/tile_datasets.py \
	-img ramp-data/PROD/ghana/images/big_image_of_ghana.tif \
	-vecs ramp-data/PROD/ghana/big_geojson_of_ghana.geojson \
	-out ramp-data/PROD/ghana/tiled \
	-pfx ghana_chip \
	-trec ramp-data/PROD/ghana/ghana_chips.geojson \
	-ndt 0.4 \
	-oht 256 \
	-owd 256
```

If you use this call, then *-out* will be the path to a directory containing two subdirectories: one named  *ramp-data/PROD/ghana/tiled/chips* and the other named *ramp-data/PROD/ghana/tiled/labels*.

### Step 3: Use the trained model to create predictions (in the form of 256x256 4-value masks) from the image chips

In this step, we run our image chips through a trained model, and write out predicted masks for each chip. The *ramp-code/scripts/get_model_predictions.py* script is used for this purpose. 

### Selecting a trained ramp model

This script requires a file path to a trained ramp model. You may have access to an existing trained ramp model provided as a zip file; if this is the case, unzipping the model will result in the creation of a directory containing several data files, and you will simply give the path to this directory.

Alternatively, you may have already trained a ramp model of your own using the 'model_checkpts' option in your training configuration file, in which case ramp model directories will have been created during your training run. The location of the model checkpoint directories is defined in your training configuration file. 

If you've been using the ramp training data setup structure described in 'How I set up my training data', and use the standard output locations for model checkpoints and logs that are described in the training configuration files, then your model checkpoints will be saved in a directory structure such as this:

```
-ramp-data
	--TRAIN
		--ghana
			--model-checkpts
				-  20220823-223408
					- model_20220823-223408_003_0.899.tf
					- model_20220823-223408_019_0.894.tf
					- model_20220823-223408_057_0.889.tf
					- etc.
```

All model checkpoints for a specific dataset (in this case, ghana) are contained in the directory *RAMP_HOME/ramp-data/ghana/model-checkpts*. They are further stored by training run, where the training run is defined by its timestamp (in this case, 20220823-223408). All the models created during that training run will be stored under its timestamp.

The model data are contained in directories with the naming format: *model_timestamp_epochnumber_accuracy.tf*. Note that these are directories and not files. Each training run will usually produce multiple models, each of which will take up quite a bit of space. 

The validation accuracy is included in the model directory names as a convenience. Frequently, you will want to select a model directory with a high validation accuracy; from the above selection, for example, you might use *model_20220823-223408_003_0.899.tf*. However, the model with the best validation accuracy might not always give the best results visually. Model prediction results for individual models can be viewed using the notebook *ramp-code/notebooks/View_predictions.ipynb*, or using the tensorboard viewing tool.

#### Running mask prediction from image chips

Given a directory containing image chips, and a path to a trained ramp model, the script is called as follows from the RAMP_HOME directory:

```
python ramp-code/scripts/get_model_predictions.py \
	-mod ramp-data/TRAIN/ghana/model-checkpts/20220823-223408/model_20220823-223408_003_0.899.tf\
	-idir ramp-data/PROD/ghana/chips \
	-odir ramp-data/PROD/ghana/20220823-223408/pred_masks 
```

All that is needed is the model path, the location of the image chips to process, and an output directory (which the script will create if it doesn't exist already). 

Important note: predicted masks depend on both the image chip set, and on the model used to produce the masks! Therefore, model prediction output is stored by both the dataset (ghana) and the model timestamp (20220823-223408). 

Unfortunately machine learning generates a lot of output, which needs to be carefully organized! 

If you are testing multiple models with the same timestamp, you may want to organize the output directories in a deeper hierarchy, as shown below:

```
ramp-data
	-PROD
		-ghana
			-chips
			-20220823-223408
				-model_20220823-223408_003_0.899
					-pred_maskss
				-model_20220823-223408_019_0.894.tf
					-pred_masks
			-20220814-164329
				-model_20220814-164329_014_0.899
					-pred_masks
				-model_20220814-164329_042_0.894.tf
					-pred_masks
			-etc
```



Output masks will have the same basename as the input chips, with the suffix *pred.tif*. The suffix *pred.tif* is used so that predicted masks will not be confused with truth masks, which are named with the suffix *mask.tif*.

### Step 4: Polygonize the 4-value masks and sew them together to create a single dataset of building models without tile boundaries

After running Step 3, you will have a directory of predicted masks that must be turned into polygons covering your area of interest. If you simply polygonize all the predicted masks, and put them together, then the resulting polygons will be broken along the edges of the tiles, which is undesirable.

The ramp codebase provides a python script, *ramp-code/scripts/get_labels_from_masks.py,* which will polygonize a set of 4-channel predicted masks in a directory, and reassemble them across tile boundaries to provide seamless building predictions.

This code is run as follows from the RAMP_HOME directory:

```
python ramp-code/scripts/get_labels_from_masks.py \
	-in ramp-data/PROD/ghana/20220823-223408/model_20220823-223408_003_0.899/pred_masks
	-out ramp-data/PROD/ghana/20220823-223408/model_20220823-223408_003_0.899/predicted_buildings.geojson
	-bwpix 2
	-crs EPSG:4326
```

- *in*: path to the directory containing the predicted masks to be processed.
- *out*: path to the output geojson file.
- *bwpix*: This parameter should be set to the width of the boundary channels in the training masks you used. The default value is 2: this is the same as the default value in the mask creation script, *ramp-code/scripts/multi_masks_from_polygons.py*.
- *crs*: Set this parameter to the EPSG code for the coordinate system you want your output in. The default value is EPSG:4326, which is lat/lon WGS84. 

Because this code fuses a large set of mask chips into a large polygon dataset, the resulting dataset may cover a large area. Because processing occurs in lat/lon coordinates, there may be distortion if the area of interest covers a very large extent from north to south, or if the AOI is well north or south of the equator. In this case, you should break it into smaller components (in the north-south direction) for production.

# Assessing the accuracy of your output polygons

## TL;DR

Assuming that you have already run the production pipeline described above for your AOIs (which is defined in the file *ramp-code/shell-scripts/run_production_on_datasets.bash*), the evaluation process described here can be run for a set of datasets at once by running edited versions of the following shell scripts in the following order:

- *ramp-code/shell-scripts/write_truth_labels_for_datasets.bash* to create fused truth polygon geojson files from truth multimasks;

- *ramp-code/shell-scripts/get_iou_metrics_for_datasets.bash* to get accuracy metrics for the fused truth and predicted polygon sets created in the last two steps.


## A note on measuring accuracy

There are different ways to measure the accuracy of the building extraction process. The measurement that occurs to most people is to measure the percentage of pixels in the image that are correctly identified as building pixels, but in cases where buildings are rare, this measurement can be very high even when no buildings at all are detected; it can therefore be misleading. However, this metric is quick to compute, and so it is one of the measurements reported by the training process during training.

Since we are interested in detecting building polygons, and not building pixels, it makes sense for us to directly measure the accuracy of our building detections. There are two questions relating to accuracy: *do we detect all the buildings that are present*, and *do we detect buildings that are not present*? The former relates to the recall metric, given by:

$$
\text{recall}=\frac{\text{number of buildings detected that match truth buildings}}{\text{number of buildings present in the truth data}}
$$

and the latter relates to precision:ramp-code/scripts/calculate_accuracy_iou.py

$$
\text{precision}=\frac{\text{number of buildings detected that match truth buildings}}{\text{number of buildings detected}}
$$

If recall is low, this means that few of the buildings present in the truth data were detected; if precision is low, few of the buildings detected were actually valid buildings. We want both of these numbers to be as high as possible.

The definitions of recall and precision depend on a notion of matching between detected buildings and truth buildings. The Spacenet Metric (which is used by ramp as its standard accuracy metric) defines two buildings as matching when their intersection-over-union (IoU) ratio is greater than 0.5 (see the image below). If more than two truth polygons match a single test (detected) polygon, the matching truth building is taken to be the one with the greater IoU value.

![[Pasted image 20220823112636.png]]

The Spacenet Metric is much more expensive to compute than the pixel-based accuracy metric, and so it is not computed during the training process. The ramp project provides scripts (derived from the [Solaris project](fill)) that computes the Spacenet Metric given any truth and test label datasets. 


## Calculating the output accuracy over a large area

In order to calculate the accuracy for a full production run, you will need the final fused-polygon output from a full production run (as described above), and a truth dataset over the same area that is also fused. The original truth labels, not the tiled truth labels, should therefore be used in this step. 

The ramp project provides the *ramp-code/scripts/calculate_accuracy_iou.py* script for this purpose. This script is called as follows from the RAMP_HOME directory:

```
python ramp-code/scripts/calculate_accuracy_iou.py
	-truth ramp-data/PROD/ghana/ghana_truth_labels.geojson
	-test ramp-data/PROD/ghana/20220823-223408/model_20220823-223408_003_0.899/predicted_buildings.geojson
	-f_m2 5.0
```

- *truth*: filepath to the geojson file containing the truth polygons
- *test*: filepath to the geojson file containing the model output to be assessed
- *f_m2*: Threshold of building area, in square meters, to be included in the assessment. If unset, the default value is 0. 

This call will produce output to the command line screen, but will not log any results.

The call below specifies that results (precision, recall, F1) be logged to a csv file. The same csv file can be logged to over and over, in order to make organizing results easier. The additional information (dataset name, model timestamp, comment) in the command line will be included in the log.

```
python ramp-code/scripts/calculate_accuracy_iou.py
	-truth ramp-data/PROD/ghana/truth_polygons.geojson
	-test ramp-data/PROD/ghana/20220823-223408/model_20220823-223408_003_0.899/predicted_buildings.geojson
	-f_m2 5.0
	-log ramp-data/PROD/all_iou_results.csv 
	-dset ghana -mid 20220823-223408 
	-note "final model"
```