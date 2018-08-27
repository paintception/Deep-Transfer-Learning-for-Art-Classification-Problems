#PLEASE IGNORE IF YOU ARE NOT RUNNING ON A GPU CLUSTER

#!/bin/bash
#SBATCH --nodes 1
#SBATCH --gres gpu:1 
#SBATCH --time=24:00:00
#SBATCH --mem=50GB

KERAS_BACKEND=tensorflow

declare -a tl_modes=("fine_tuning" "off_the_shelf" "random") #The three pre-training approaches investigated in the paper

dataset_name="" #The name of your dataset

metadata_path="" #The path to your metadata file in .csv extension
jpg_images_path="" #The path to your images in *.jpg

results_path="" #The path where wou would like to store your results
datasets_path="" #The path where the hdf5 files will be stored for the experiments

tl_mode="" #Choose a pre-training mode 

python transfer_learning_experiment.py --dataset_name $dataset_name --ANN "ResNet" --metadata_path $metadata_path --jpg_images_path $jpg_images_path --results_path $results_path --datasets_path $datasets_path --tl_mode $tl_mode 
