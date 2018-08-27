#!/bin/bash
#SBATCH --nodes 1
#SBATCH --gres gpu:1 
#SBATCH --time=24:00:00
#SBATCH --mem=50GB

source activate PhD
KERAS_BACKEND=tensorflow

declare -a tl_modes=("fine_tuning" "off_the_shelf" "random")

dataset_name="a"

metadata_path="/home/matthia/Desktop/data_subset/data_subset.csv"
jpg_images_path="/home/matthia/Desktop/data_subset/"

results_path="/home/matthia/Desktop/"
datasets_path="/home/matthia/Desktop/"

tl_mode="fine_tuning"

python transfer_learning_experiment.py --dataset_name $dataset_name --ANN "ResNet" --metadata_path $metadata_path --jpg_images_path $jpg_images_path --results_path $results_path --datasets_path $datasets_path --tl_mode $tl_mode 
