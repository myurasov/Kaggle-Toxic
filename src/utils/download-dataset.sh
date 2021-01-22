#!/bin/sh

DATA_DIR="/app/_data"

rm -rfv "${DATA_DIR}/src"
mkdir -pv "${DATA_DIR}/src"
cd "${DATA_DIR}/src"

kaggle competitions download -c jigsaw-toxic-comment-classification-challenge

unzip jigsaw-toxic-comment-classification-challenge.zip
rm jigsaw-toxic-comment-classification-challenge.zip

unzip "*.zip"
rm -v *.zip