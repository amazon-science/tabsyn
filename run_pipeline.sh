#!/bin/bash

#------------------
# First active conda environment, and then navigate to the Tabsyn directory, and run this file
# 
# conda activate tabsyn
# cd /home/ajay/Python_Projects/tabsyn-main/
# ./run_pipeline.sh
#
#------------------
# Problems?
#
# Make sure this file is saved in the tabsyn directory as run_pipeline.sh
#
# Also that it is executable, use
#
# chmod +x run_pipeline.sh
#
##==================================================
# For tensorboard do
# conda activate tabsyn
# cd /home/ajay/Python_Projects/tabsyn-main/
# 
# tensorboard --logdir=vae_runs
#
# tensorboard --logdir=diffusion_runs
##==================================================

#!/bin/bash

# Set experiment name once
EXPERIMENT_NAME="experiment_13"

# Process data
#echo "Processing data..."
#python process_dataset.py --dataname "$EXPERIMENT_NAME"
#if [ $? -ne 0 ]; then
#    echo "Error in Processing data. Exiting."
#    exit 1
#fi

# Train VAE
#echo "Running VAE training..."
#python main.py --dataname "$EXPERIMENT_NAME" --method vae --mode train
#if [ $? -ne 0 ]; then
#    echo "Error in VAE training. Exiting."
#    exit 1
#fi

# Train TabSyn
echo "Running TabSyn training..."
python main.py --dataname "$EXPERIMENT_NAME" --method tabsyn --mode train
if [ $? -ne 0 ]; then
    echo "Error in TabSyn training. Exiting."
    exit 1
fi

# Multiple Sampling - make sure run_sampling_script.py is correctly specified
echo "Running Sampling script..."
python run_sampling_script.py --dataname "$EXPERIMENT_NAME"
if [ $? -ne 0 ]; then
    echo "Error in Sampling. Exiting."
    exit 1
fi

echo "All steps completed successfully!"

# Sample - single sample not so useful
#echo "Running Sample..."
#python main.py --dataname experiment_9 --method tabsyn --mode sample --save_path synthetic/experiment_9/tabsyn.csv
#if [ $? -ne 0 ]; then
#    echo "Error in Sample. Exiting."
#    exit 1
#fi

## Impute - see the algorithm in the paper, algorithm 4, page 25, this is NOT sampling
#echo "Running Impute..."
#python impute.py --dataname experiment_9 --num_steps 50 --num_samples 100
#if [ $? -ne 0 ]; then
#    echo "Error in Impute. Exiting."
#    exit 1
#fi

