#
# Usage: ./batch_compress_geotiffs.sh -i INPUT_FOLDER_PATH -o OUTPUT_FOLDER_PATH  
#test if copy 24082025

set -e

while getopts i:o: flag
do
    case "${flag}" in
        i) input=${OPTARG};;
        o) output=${OPTARG};;
    esac
done

echo "Input: $input"
echo "Ouput: $output"

counter=0
for file in "$input"/*.tif; 
do 
  let counter=counter+1
  filename=$(basename -- "$file")
  # filename="${filename*.}"
  echo "Compressing $file to $filename ..."
  gdal_translate -of COG -co COMPRESS=JPEG -co QUALITY=50 "$file" "$output"/"$filename" &
  
  if [ $counter -ge 25 ]; then
    echo "Waiting to complete current batch..."
    wait;
    let counter=0
  fi
done
wait;

