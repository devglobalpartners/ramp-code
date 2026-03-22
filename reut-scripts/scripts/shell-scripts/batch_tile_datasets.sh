#
# Usage: ./batch_tile_datasets.sh -i INPUT_FOLDER_PATH -o OUTPUT_FOLDER_PATH -p CHIP_PREFIX 
#

while getopts i:o:p: flag
do
    case "${flag}" in
        i) input=${OPTARG};;
        o) output=${OPTARG};;
        p) prefix=${OPTARG};;
    esac
done

echo "Input: $input"
echo "Ouput: $output"
echo "Prefix: $prefix"

for file in "$input"/*.tif; 
do 
  echo "Processing $file ..."
  python ./ramp-code/scripts/tile_datasets.py -img "$file" -out "$output" -pfx "$prefix" -ndt 0.4 -oht 256 -owd 256 ; 
done

