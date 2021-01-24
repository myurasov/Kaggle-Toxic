# Code for Toxic Comment Classification Challenge

https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/

## Setup

### Download Dataset

`$ docker/docker.sh "src/download_dataset.sh"`

`kaggle.json` with valid API token needs to be placed in the application directory prior to downloading data or uploading submissions.

### Download Pre-trained BERT

`$ docker/docker.sh "src/download_bert.sh"`

### Prepare Dataset for BERT

`$ docker/docker.sh "src/preprocess_for_bert.py"`

### Train BERT-based Classifier

`$ docker/docker.sh "src/train_bert.py [--run] [--epochs] [--warmup_epochs] [--batch] [--max_items]"`

### Generating submission with BERT-based Classifier

`$ docker/docker.sh "src/infer_bert.py  [--model_file] [--submission_file] [--batch] [--max_items]"`

## Starting Jupyter Lab and TensorBoard

`$ docker/docker-forever.sh [--jupyter_port=####|8888] [--tensorboard_port=####|6006]`

## @see

 - https://github.com/google-research/bert
 - https://arxiv.org/pdf/1810.04805.pdf - BERT
 - https://arxiv.org/pdf/1902.00751.pdf - Adapter-BERT
 - https://arxiv.org/pdf/1909.11942.pdf - ALBERT