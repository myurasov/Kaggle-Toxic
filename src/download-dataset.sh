#!/bin/sh

DATA_DIR=`python -c "from src.config import config; print(config['DATA_DIR'])"`
DEST_DIR="${DATA_DIR}/src"

rm -rfv $DEST_DIR
mkdir -pv $DEST_DIR
cd $DEST_DIR

kaggle competitions download -c jigsaw-toxic-comment-classification-challenge

unzip jigsaw-toxic-comment-classification-challenge.zip
rm jigsaw-toxic-comment-classification-challenge.zip

unzip "*.zip"
rm -v *.zip