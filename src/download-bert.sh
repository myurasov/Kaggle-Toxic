#!/bin/sh

# download pre-trained BERT from gcloud

DATA_DIR=`python -c "from src.config import config; print(config['DATA_DIR'])"`
DEST_DIR="${DATA_DIR}/bert"

MODEL_DIR="2018_10_18"
MODEL_NAME="uncased_L-12_H-768_A-12"

rm -rfv $DEST_DIR
mkdir -pv $DEST_DIR
cd $DEST_DIR

for f in "bert_config.json" "vocab.txt" "bert_model.ckpt.meta" \
    "bert_model.ckpt.index" "bert_model.ckpt.data-00000-of-00001"
do
    cmd="gsutil cp gs://bert_models/${MODEL_DIR}/${MODEL_NAME}/${f} $DEST_DIR/${f}"
    $cmd
done
