#!/bin/bash
# MLOps stack installation script
# This script does NOT install: Prometheus

# -----------------------
# Misc tools
# -----------------------

git lfs install
apt-get install git
sudo apt-get install -y adduser libfontconfig1
wget https://dl.grafana.com/oss/release/grafana_8.5.5_amd64.deb
sudo dpkg -i grafana_8.5.5_amd64.deb

# -----------------------
# Python packages
# -----------------------

pip install mlflow
pip install jupyterlab
pip install prefect
pip install fastapi
pip install evidently