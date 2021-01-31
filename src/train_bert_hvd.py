#!/usr/bin/python

import math

import horovod.tensorflow.keras as hvd
import tensorflow as tf
from lib.bert_utils import (
    bert_build_model,
    bert_get_training_arguments,
    bert_load_training_data,
)
from lib.train_utils import create_tensorboard_run_dir, save_trained_model
from tensorflow import keras

from src.config import config

# read cli arguments
args = bert_get_training_arguments("BERT Classifier, version A")

# Horovod: init
hvd.init()

# Horovod: Pin GPU to be used to process local rank (one GPU per process)
gpus = tf.config.experimental.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
if gpus:
    tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], "GPU")
print(f"* Horovod: Using {len(gpus)} GPU(s)")

# Horovod: adjust number of epochs based on number of GPUs.
args.epochs = int(math.ceil(args.epochs / hvd.size()))
print(f"* Horovod: args.epochs adjusted to {args.epochs}")

# Horovod: adjust learning rate based on number of GPUs.
args.lr_start *= hvd.size()
print(f"* Horovod: args.lr_start adjusted to {args.lr_start}")

# prepare model

model = bert_build_model(
    bert_model_dir=config["DATA_DIR"] + "/bert", max_seq_len=config["MAX_SEQ_LENGTH"]
)

optimizer = keras.optimizers.Adam(learning_rate=args.lr_start)

# Horovod: add Horovod Distributed Optimizer
optimizer = hvd.DistributedOptimizer(optimizer)

model.compile(
    optimizer=optimizer,
    loss=keras.losses.BinaryCrossentropy(),
    metrics=[
        keras.metrics.AUC(
            multi_label=True,
        )
    ],
    # Horovod: Specify `experimental_run_tf_function=False` to ensure TensorFlow
    # uses hvd.DistributedOptimizer() to compute gradients.
    experimental_run_tf_function=False,
)

model.summary()

#

# load training data
train_X, train_Y, val_X, val_Y = bert_load_training_data(
    max_items=args.max_items, shuffle=True, val_split=args.val_split
)

# create tensorboard log dir
tb_log_dir = create_tensorboard_run_dir(args.run)

# fit

model.fit(
    x=train_X,
    y=train_Y,
    shuffle=True,
    epochs=args.epochs,
    batch_size=args.batch,
    validation_data=(val_X, val_Y),
    steps_per_epoch=args.samples_per_epoch // args.batch,
    callbacks=[
        # Horovod: broadcast initial variable states from rank 0 to all other processes.
        # This is necessary to ensure consistent initialization of all workers when
        # training is started with random weights or restored from a checkpoint.
        hvd.callbacks.BroadcastGlobalVariablesCallback(0),
        #
        keras.callbacks.EarlyStopping(
            patience=args.early_stop_patience, restore_best_weights=True, verbose=1
        ),
        keras.callbacks.TensorBoard(log_dir=tb_log_dir),
    ],
)

# # Horovod: save checkpoints only on worker 0 to prevent other workers from corrupting them.
# if hvd.rank() == 0:
#     model.callbacks.append(keras.callbacks.ModelCheckpoint("./checkpoint-{epoch}.h5"))

# save trained model
save_trained_model(model=model, run=args.run, info=vars(args))
