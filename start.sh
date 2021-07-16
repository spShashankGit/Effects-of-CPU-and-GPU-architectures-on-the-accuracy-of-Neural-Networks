#!/bin/bash

# Directory name
APP_NAME="thesis_experiment"

cd
mkdir $APP_NAME
cd $APP_NAME

# Create and activate conda env with with python 3.6
#conda create -n thesisExpCondaEnv python=3.6
#conda activate thesisExpCondaEnv 

conda create -n thesisExpCondaEnv_P38 python=3.8
conda activate thesisExpCondaEnv_P38

# Install conda packages from the req file
conda install --file requirements_gpuInfoLogger1.txt -y

# Install pycuda via pip
python3 -m pip install pycuda==2020.1

# Install pytorch 
conda install -c pytorch pytorch 

# Set environment variables for pypads logging
export MONGO_DB=pypads;
export MONGO_USER=pypads;
export MONGO_URL=mongodb://ilz.dimis.fim.uni-passau.de:29642;
export MONGO_PW=8CN7OqknwhYr3RO;

# Steps to run the experiment
#python3 -m pip install Code/PyPads_GPU_Details_Logger/dist/pypads-0.5.7_with_env_variable.tar.gz
#python3 Code/Logistic_Map/Logistic_Map_Script.py
#python3 Code/VGG-11/vgg_numpy.py

# Now install pypads from dist
echo "Script completed"