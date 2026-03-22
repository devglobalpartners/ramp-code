### RUN THIS IN YOUR CONTAINER AFTER ANY UPDATES

echo ================ NVIDIA-SMI check ================
nvidia-smi


echo
echo ================ Tensorflow check ================
echo 
TF_CPP_MIN_LOG_LEVEL=3  python -c 'import tensorflow as tf; print("\nTensorflow version: ",tf.__version__, "\nTensorflow file: ",tf.__file__) ;  print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices("GPU")))'