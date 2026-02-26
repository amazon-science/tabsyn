#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Assuming this script is saved in the directory, 

/home/ajay/Python_Projects/My_TabSyn_Experiments

This script can be run from the terminal/command line using

cd /home/ajay/Python_Projects/My_TabSyn_Experiments
python run_sampling_script.py

"""

import os

def run_sampling_repeatedly(dataname, method, mode, save_dir, num_samples):
    """
    Automates running the command-line sampling tool multiple times.

    Parameters:
        dataname (str): Name of the dataset (e.g., 'experiment_9').
        method (str): Sampling method (e.g., 'tabsyn').
        mode (str): Mode of execution (e.g., 'sample').
        save_dir (str): Directory to save the output files.
        num_samples (int): Number of samples to generate.
    """
    # Ensure save directory exists
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # Loop to run the sampling command
    for i in range(num_samples):
        save_path = os.path.join(save_dir, f"sample_{i}.csv")
        command = (
            f"python main.py --dataname {dataname} "
            f"--method {method} --mode {mode} --save_path {save_path}"
        )
        print(f"Running command: {command}")
        os.system(command)  # Run the command

    print(f"Successfully generated {num_samples} samples in {save_dir}.")

# Example usage
if __name__ == "__main__":
    DATANAME = "experiment_9"  # Dataset name
    METHOD = "tabsyn"          # Sampling method
    MODE = "sample"            # Mode of execution
    SAVE_DIR = "/home/ajay/Python_Projects/My_TabSyn_Experiments/sampled_OA_level_data/experiment_9"
    NUM_SAMPLES = 3           # Number of samples to generate

    # Run the function
    run_sampling_repeatedly(DATANAME, METHOD, MODE, SAVE_DIR, NUM_SAMPLES)
