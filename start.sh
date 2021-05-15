#!/bin/bash

# Directory name
APP_NAME="thesis_experiment"

cd
mkdir $APP_NAME
cd $APP_NAME

# Create and activate conda env with with python 3.6
conda create -n thesisExpCondaEnv python=3.6
conda activate thesisExpCondaEnv 

# Install conda packages from the req file
conda install --file requirements_gpuInfoLogger1.txt -y

# Install pycuda via pip
python3 -m pip install pycuda==2020.1

# Set environment variables for pypads logging
export MONGO_DB=pypads
export MONGO_USER=pypads
export MONGO_URL=mongodb://www.padre-lab.eu:2222
export MONGO_PW=8CN7OqknwhYr3RO

echo "Script finished"