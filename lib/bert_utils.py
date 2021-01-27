import argparse
import math
from pathlib import Path
from pprint import pformat

import bert
import numpy as np
from src.config import config
from tensorflow import keras


def bert_preprocess_text(tokenizer, text, max_text_len, max_seq_len):
    """
    Convert piece of text into tokenized version for bert
    """

    # tokenize
    tokens = tokenizer.tokenize(text[:max_text_len])
    tokens = tokens[: max_seq_len - 2]
    tokens = ["[CLS]"] + tokens + ["[SEP]"]
    token_ids = tokenizer.convert_tokens_to_ids(tokens)

    assert len(token_ids) <= max_seq_len

    # pad
    token_ids += [0] * (max_seq_len - len(token_ids))

    assert len(token_ids) == max_seq_len
    return token_ids


def bert_build_model(bert_model_dir, max_seq_len):
    """
    Build multi-class classifier model based on pre-trained bert
    """

    # create bert layer
    bert_config_path = bert_model_dir + "/bert_config.json"
    bert_ckpt_path = bert_model_dir + "/bert_model.ckpt"
    bert_config = Path(bert_config_path).read_text()
    bert_config = bert.loader.StockBertConfig.from_json_string(bert_config)
    bert_params = bert.loader.map_stock_config_to_params(bert_config)
    bert_params.adapter_size = None  # enable full retraining
    bert_layer = bert.BertModelLayer.from_params(bert_params, name="bert_layer")

    # input for token ids of (,MAX_SEQ_LENGTH) shape
    input_token_ids = keras.layers.Input(
        shape=(max_seq_len,), dtype="int32", name="input_token_ids"
    )

    # bert body
    x = bert_layer(input_token_ids)

    #  classification head
    x = keras.layers.Lambda(lambda seq: seq[:, 0, :])(x)
    x = keras.layers.Dropout(0.5)(x)
    x = keras.layers.Dense(units=768, activation="relu")(x)
    x = keras.layers.Dropout(0.5)(x)

    # output
    # sigmoid is preferred for mult-class classification
    output = keras.layers.Dense(units=6, activation="sigmoid")(x)

    model = keras.Model(inputs=[input_token_ids], outputs=[output])
    model.build(input_shape=(None, max_seq_len))

    # load the pre-trained model weights
    bert.loader.load_stock_weights(bert_layer, bert_ckpt_path)

    return model


def bert_create_lr_scheduler(
    max_learn_rate=1e-5,
    end_learn_rate=1e-7,
    warmup_epochs=20,
    epochs_total=50,
    horovod_factor=1.0,  # if using horovod, lr should be multiplied by number of GPUs
):
    """
    Create LR scheduler for BERT-based multiclass classifier
    @see https://www.desmos.com/calculator/klvwfuidie
    """

    def _lr_scheduler(epoch):
        if epoch < warmup_epochs:
            res = (max_learn_rate / warmup_epochs) * (epoch + 1)
        else:
            res = max_learn_rate * math.exp(
                math.log(end_learn_rate / max_learn_rate)
                * (epoch - warmup_epochs + 1)
                / (epochs_total - warmup_epochs + 1)
            )
        return float(res)

    return keras.callbacks.LearningRateScheduler(_lr_scheduler, verbose=1)


# read cli arguments for bert training
def bert_get_training_arguments(
    description="Train BERT-based classifier",
    RUN="A",
    LR_END=5e-8,
    LR_START=5e-6,
    VAL_SPLIT=0.1,
    BATCH_SIZE=48,
    TOTAL_EPOCHS=50,
    WARMUP_EPOCHS=10,
    EARLY_STOP_PATIENCE=10,
):

    parser = argparse.ArgumentParser(
        description=description,
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

    # display arguments
    print(f"* Arguments:\n{pformat(vars(args))}")

    return args


# load training data
def bert_load_training_data(max_items=None):

    train_X = np.load(config["DATA_DIR"] + "/processsed_for_bert/train.X.npy").astype(
        np.int32
    )
    train_Y = np.load(config["DATA_DIR"] + "/processsed_for_bert/train.Y.npy").astype(
        np.float32
    )

    # limit max dataset size
    if max_items is not None:

        # shuffle before limiting
        indexes = np.random.permutation(len(train_X))
        train_X = train_X[indexes]
        train_Y = train_Y[indexes]

        train_X = train_X[:max_items]
        train_Y = train_Y[:max_items]

    return train_X, train_Y
