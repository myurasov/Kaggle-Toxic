#!/usr/bin/python

import argparse
import os
import shutil
from datetime import datetime
from pprint import pformat, pprint

import numpy as np
from lib.bert_utils import (
    build_bert_classifier_model,
    create_bert_learning_rate_scheduler,
)
from tensorflow import keras

from src.config import config

# settings

RUN = "A"
LR_END = 5e-8
LR_START = 5e-6
VAL_SPLIT = 0.1
BATCH_SIZE = 48
TOTAL_EPOCHS = 50
WARMUP_EPOCHS = 10
EARLY_STOP_PATIENCE = 10

# read cli arguments

parser = argparse.ArgumentParser(
    description="Train BERT-based classifier",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)

parser.add_argument("--run", type=str, default=RUN)
parser.add_argument("--max_items", type=int, default=None)
parser.add_argument("--epochs", type=int, default=TOTAL_EPOCHS)
parser.add_argument("--warmup_epochs", type=int, default=WARMUP_EPOCHS)
parser.add_argument("--batch", type=int, default=BATCH_SIZE)
parser.add_argument("--lr_start", type=int, default=LR_START)
parser.add_argument("--lr_end", type=int, default=LR_END)
parser.add_argument("--val_split", type=float, default=VAL_SPLIT)
parser.add_argument("--early_stop_patience", type=int, default=EARLY_STOP_PATIENCE)

args = parser.parse_args()

print("* Arguments: ")
pprint(vars(args))

# prepare model

model = build_bert_classifier_model(
    bert_model_dir=config["DATA_DIR"] + "/bert", max_seq_len=config["MAX_SEQ_LENGTH"]
)

model.compile(
    optimizer=keras.optimizers.Adam(),
    loss=keras.losses.BinaryCrossentropy(),
    metrics=[
        keras.metrics.AUC(
            multi_label=True,
        )
    ],
)

# load training data
train_X = np.load(config["DATA_DIR"] + "/processsed_for_bert/train.X.npy").astype(
    np.int32
)
train_Y = np.load(config["DATA_DIR"] + "/processsed_for_bert/train.Y.npy").astype(
    np.float32
)

# limit max dataset size
if args.max_items is not None:

    # shuffle before limiting
    indexes = np.random.permutation(len(train_X))
    train_X = train_X[indexes]
    train_Y = train_Y[indexes]

    train_X = train_X[: args.max_items]
    train_Y = train_Y[: args.max_items]

# tensorboard log dir
tb_log_dir = f"/app/.tensorboard/{args.run}"
shutil.rmtree(tb_log_dir, ignore_errors=True)

# fit
model.fit(
    x=train_X,
    y=train_Y,
    shuffle=True,
    epochs=args.epochs,
    batch_size=args.batch,
    validation_split=args.val_split,
    callbacks=[
        create_bert_learning_rate_scheduler(
            max_learn_rate=args.lr_start,
            end_learn_rate=args.lr_end,
            warmup_epochs=args.warmup_epochs,
            epochs_total=args.epochs,
        ),
        keras.callbacks.EarlyStopping(
            patience=args.early_stop_patience, restore_best_weights=True, verbose=1
        ),
        keras.callbacks.TensorBoard(log_dir=tb_log_dir),
    ],
)

# save trained model

output_dir = config["DATA_DIR"] + "/saved_models"
model_path = output_dir + f"/model_bert.{args.run}.tf"
os.makedirs(output_dir, exist_ok=True)
model.save(model_path, overwrite=True, save_format="tf", include_optimizer=True)
print(f"* Model saved to {model_path}")

# save run info for later reference

info_path = model_path + "_info.txt"
print(
    f"Date:\n\n{datetime.now()} UTC\n\nArguments:\n\n{pformat(vars(args))}",
    file=open(info_path, "w"),
)
