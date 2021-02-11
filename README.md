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

`$ docker/docker.sh "src/train_bert-[variant].py [arguments]"`

Options available:

<pre>
-h, --help
--run RUN
--max_items MAX_ITEMS
--epochs EPOCHS
--batch BATCH
--lr_start LR_START
--val_split VAL_SPLIT
--early_stop_patience EARLY_STOP_PATIENCE
--samples_per_epoch SAMPLES_PER_EPOCH
</pre>

To run with Horovod:

`$ docker/docker.sh "src/train_bert-[variant].py [arguments]"`

### Generating submission with BERT-based Classifier

`$ docker/docker.sh --gpus='"device=0,1,2,###"' "horovodrun -np ### src/train_bert_hvd.py"`

Options available:

<pre>
-h, --help
--model_file MODEL_FILE
--submission_file SUBMISSION_FILE
--batch BATCH
--max_items MAX_ITEMS
</pre>

## Starting Jupyter Lab and TensorBoard

`$ docker/docker-forever.sh [--jupyter_port=####|8888] [--tensorboard_port=####|6006]`

## @see

 - https://github.com/google-research/bert
 - https://arxiv.org/pdf/1810.04805.pdf - BERT
 - https://arxiv.org/pdf/1902.00751.pdf - Adapter-BERT
 - https://arxiv.org/pdf/1909.11942.pdf - ALBERT