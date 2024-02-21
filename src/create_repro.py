import os, sys
import subprocess
from pathlib import Path

if __name__ == "__main__":
    '''
    Creates the basic data structure in /data to start reproducing results from scratch, 
    thereby creating another sample
    '''

    # create directory structure
    model_path = Path("../data/repro/models")
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    figure_path = Path("../data/repro/figures")
    if not os.path.exists(figure_path):
        os.makedirs(figure_path)

    exp_path = Path("../data/repro/experiments")
    if not os.path.exists(exp_path):
        os.makedirs(exp_path)

    # learn bio-layer on CIFAR10 data according to Krotov and Hopfield
    fkhl3_name = Path("fkhl3_cifar10.pty")
    subprocess.call(
        "python ./llearn_CIFAR.py " + str(model_path / fkhl3_name),
        shell=True,
        )

    # prune the previously learned model and create Figure A1
    subprocess.call(
        "python ./prune_and_plot_FKHL3_CIFAR.py " + str(model_path / fkhl3_name) + " " + str(figure_path),
        shell=True,
        )
