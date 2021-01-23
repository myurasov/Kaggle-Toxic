# Code for Toxic Comment Classification Challenge

https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/

## Setup

### Dataset Download

`$ docker/docker.sh "src/utils/download-dataset.sh"`

`kaggle.json` with valid API token needs to be placed in the application directory prior to downloading data or uploading submissions.

### Download Pre-trained BERT

`$ docker/docker.sh "src/utils/download-bert.sh"`

## Starting Jupyter Lab and TensorBoard

`$ docker/docker-forever.sh [--jupyter_port=####|8888] [--tensorboard_port=####|6006]`
