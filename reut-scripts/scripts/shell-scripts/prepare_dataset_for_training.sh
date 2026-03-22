#
# Usage: ./prepare_dataset_for_training.sh -i INPUT_FOLDER_PATH  
#

while getopts i: flag
do
    case "${flag}" in
        i) input=${OPTARG};;
    esac
done

echo "Input: $input"


# Create masks from labels
python /tf/ramp-code/scripts/multi_masks_from_polygons.py -in_vecs /tf/"$input"/labels -in_chips /tf/"$input"/source -out /tf/"$input"/multimasks

# Split training and validation data - create csv list
python /tf/ramp-code/scripts/make_train_val_split_lists.py -src /tf/"$input"/source -pfx dataset_split -trn 0.85 -val 0.15 

# Move all the TRAINING chips to their new locations 
python /tf/ramp-code/scripts/move_chips_from_csv.py -sd /tf/"$input"/source -td /tf/"$input"/train/chips -csv /tf/"$input"/dataset_split_train.csv -mv 

python /tf/ramp-code/scripts/move_chips_from_csv.py -sd /tf/"$input"/labels -td /tf/"$input"/train/labels -csv /tf/"$input"/dataset_split_train.csv -mv 

python /tf/ramp-code/scripts/move_chips_from_csv.py -sd /tf/"$input"/multimasks -td /tf/"$input"/train/multimasks -csv /tf/"$input"/dataset_split_train.csv -mv 

# Move all the VALIDATION chips to their new locations 
python /tf/ramp-code/scripts/move_chips_from_csv.py -sd /tf/"$input"/source -td /tf/"$input"/validate/chips -csv /tf/"$input"/dataset_split_val.csv -mv 

python /tf/ramp-code/scripts/move_chips_from_csv.py -sd /tf/"$input"/labels -td /tf/"$input"/validate/labels -csv /tf/"$input"/dataset_split_val.csv -mv 

python /tf/ramp-code/scripts/move_chips_from_csv.py -sd /tf/"$input"/multimasks -td /tf/"$input"/validate/multimasks -csv /tf/"$input"/dataset_split_val.csv -mv 

