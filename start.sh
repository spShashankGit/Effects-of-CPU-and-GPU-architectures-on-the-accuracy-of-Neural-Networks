#!/bin/bash

# Directory name
# APP_NAME="thesis_experiment_shashank"

# cd
# mkdir $APP_NAME
# cd $APP_NAME

# Create and activate conda env with with python 3.6
#conda create -n thesisExpCondaEnv python=3.6
#conda activate thesisExpCondaEnv 

#conda create -n thesisExpCondaEnv_P38 python=3.8
#conda activate thesisExpCondaEnv_P38

# Install conda packages from the req file
#conda install --file requirements_gpuInfoLogger1.txt -y
pip3 install -r requirements_gpuInfoLogger1.txt


# Install pycuda via pip
python3 -m pip3 install pycuda==2020.1


# Install pytorch 
pip3 install torch 


# Set environment variables for pypads logging
export MONGO_DB=pypads;
export MONGO_USER=pypads;
export MONGO_URL=mongodb://ilz.dimis.fim.uni-passau.de:29642;
export MONGO_PW=8CN7OqknwhYr3RO;


# Update to Python 3.8.0
apt update
sudo apt install software-properties-common
add-apt-repository ppa:deadsnakes/ppa
apt update
apt install python3.8
python3 --version

#.bashrc
#alias="/usr/bin/python3.8"
#https://medium.com/analytics-vidhya/installing-python-3-8-3-66701d3db134


# Install pypads
python3 -m pip3 install Code/PyPads_GPU_Details_Logger/dist/pypads-0.5.7_with_env_variable.tar.gz


# Check if all packages are installed correctly
pip3 freeze >> pipFreeze.txt
python3 compare.py


# Steps to run the experiment
# python3 Code/Logistic_Map/Logistic_Map_Script.py
# python3 Code/VGG-11/vgg_numpy.py


# Now install pypads from dist
echo "Script completed"