# Code for Toxic Comment Classification Challenge

https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/

## Setup

### Download Dataset

`$ docker/docker.sh "src/download-dataset.sh"`

`kaggle.json` with valid API token needs to be placed in the application directory prior to downloading data or uploading submissions.

### Download Pre-trained BERT

`$ docker/docker.sh "src/download-bert.sh"`

### Prepare Dataset for BERT

`$ docker/docker.sh "python src/preprocess_for_bert.py"`

## Starting Jupyter Lab and TensorBoard

`$ docker/docker-forever.sh [--jupyter_port=####|8888] [--tensorboard_port=####|6006]`
