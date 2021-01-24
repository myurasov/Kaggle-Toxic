#!/usr/bin/python

import os
import shutil

import numpy as np
from lib.bert_utils import (
    build_bert_classifier_model,
    create_bert_learning_rate_scheduler,
)
from tensorflow import keras

from src.config import config


RUN = "A"
LR_END = 1e-7
LR_START = 1e-5
BATCH_SIZE = 48
TOTAL_EPOCHS = 50
WARMUP_EPOCHS = 20
VALIDATION_SPLIT = 0.1
EARLY_STOP_PATIENCE = 20

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

# load trainign data
train_X = np.load(config["DATA_DIR"] + "/processsed_for_bert/train.X.npy")
train_Y = np.load(config["DATA_DIR"] + "/processsed_for_bert/train.Y.npy")
train_X, train_Y = train_X[:48], train_Y[:48]

# tensorboard log dir
tb_log_dir = f"/app/.tensorboard/{RUN}"
shutil.rmtree(tb_log_dir, ignore_errors=True)

# fit
model.fit(
    x=train_X,
    y=train_Y,
    validation_split=VALIDATION_SPLIT,
    batch_size=BATCH_SIZE,
    epochs=TOTAL_EPOCHS,
    shuffle=True,
    callbacks=[
        create_bert_learning_rate_scheduler(
            max_learn_rate=LR_START,
            end_learn_rate=LR_END,
            warmup_epochs=WARMUP_EPOCHS,
            epochs_total=TOTAL_EPOCHS,
        ),
        keras.callbacks.EarlyStopping(
            patience=EARLY_STOP_PATIENCE, restore_best_weights=True
        ),
        keras.callbacks.TensorBoard(log_dir=tb_log_dir),
    ],
)

# save trained model

output_dir = config["DATA_DIR"] + "/saved_models_bert"
os.makedirs(output_dir, exist_ok=True)
model.save(output_dir + "/model.h5", overwrite=True)
