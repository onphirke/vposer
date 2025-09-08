#!/usr/bin/env bash

# install system dependencies
sudo apt-get update
sudo apt-get install libgl1-mesa-glx libglu1-mesa freeglut3-dev

# check that we have uv installed
if ! command -v uv &> /dev/null
then
    echo "uv could not be found, please install it first"
    exit
fi

# if there is no .venv, create one
if [ ! -d ".venv" ]; then
    uv venv .venv --python 3.8
    source .venv/bin/activate
    uv pip install -r requirements.txt

    # fix pyopengl deps
    uv pip install PyOpenGL --reinstall --no-cache-dir

    # install this package in develop mode
    uv run setup.py develop
fi

