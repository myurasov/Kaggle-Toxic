#!/bin/bash

# download pre-trained BERT from gcloud

DATA_DIR=`python -c "import src.config import config as c; print(config['DATA_DIR'])"`
MODEL_PATH=`python -c "from src.config import config; print(config['GS_BERT_MODEL_PATH'])"`
DEST_DIR="${DATA_DIR}/bert"

rm -rfv $DEST_DIR
mkdir -pv $DEST_DIR
cd $DEST_DIR

for f in "bert_config.json" "vocab.txt" "bert_model.ckpt.meta" \
    "bert_model.ckpt.index" "bert_model.ckpt.data-00000-of-00001"
do
    cmd="gsutil cp gs://bert_models/${MODEL_PATH}/${f} $DEST_DIR/${f}"
    $cmd
done
