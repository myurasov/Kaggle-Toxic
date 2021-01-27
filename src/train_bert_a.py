#!/usr/bin/python

from lib.bert_utils import (
    bert_build_model,
    bert_create_lr_scheduler,
    bert_get_training_arguments,
    bert_load_training_data,
)
from lib.train_utils import create_tensorboard_run_dir, save_trained_model
from tensorflow import keras

from src.config import config

# read cli arguments
args = bert_get_training_arguments("BERT Classifier, version A")

# create tensorboard log dir
tb_log_dir = create_tensorboard_run_dir(args.run)

# load training data
train_X, train_Y = bert_load_training_data(args.max_items)

# prepare model

model = bert_build_model(
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

model.summary()

# fit

model.fit(
    x=train_X,
    y=train_Y,
    shuffle=True,
    epochs=args.epochs,
    batch_size=args.batch,
    validation_split=args.val_split,
    callbacks=[
        bert_create_lr_scheduler(
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
save_trained_model(model=model, run=args.run, info=vars(args))
