#!/usr/bin/python

import argparse
import csv
import os

import numpy as np
from tensorflow import keras

from src.config import config

# settings

BATCH_SIZE = 48

# read cli arguments

parser = argparse.ArgumentParser(
    description="Infer BERT-based classifier and create Kaggle submission",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)

parser.add_argument("--model_file", type=str, default="model_bert.A.tf")
parser.add_argument("--submission_file", type=str, default="submission.A.csv")
parser.add_argument("--batch", type=int, default=BATCH_SIZE)
parser.add_argument("--max_items", type=int, default=None)

args = parser.parse_args()
print("* Using arguments: ", args)

# prepare model
args.model_file = config["DATA_DIR"] + "/saved_models/" + args.model_file
print(f"* Loading model from {args.model_file} ...")
model = keras.models.load_model(args.model_file)
model.summary()

# load test data
test_ids = np.load(config["DATA_DIR"] + "/processsed_for_bert/test.ids.npy")
test_X = np.load(config["DATA_DIR"] + "/processsed_for_bert/test.X.npy").astype(
    np.int32
)

# limit max dataset size
if args.max_items is not None:
    test_X = test_X[: args.max_items]

# predict

print(f"* Predicting on {len(test_X)} items...")
test_Y = model.predict(test_X, batch_size=args.batch, verbose=1)

# generate CSV
output_dir = config["DATA_DIR"] + "/submissions"
os.makedirs(output_dir, exist_ok=True)
args.submission_file = output_dir + "/" + args.submission_file
print(f"* Writing to {args.submission_file} ...")

with open(args.submission_file, "w") as csvfile:
    csv_writer = csv.writer(csvfile, delimiter=",")

    csv_writer.writerow(
        ["id", "toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
    )

    for i in range(0, len(test_X)):
        csv_writer.writerow([test_ids[i]] + list(test_Y[i]))
