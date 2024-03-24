#!/usr/bin/env bash
echo "Installing environment:"
conda env create -f environment.yml
echo "All libs were installed, please activate env by running: conda activate env_name"
