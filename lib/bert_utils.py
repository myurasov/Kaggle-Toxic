from pathlib import Path

import bert
from tensorflow import keras


def preprocess_text_for_bert(tokenizer, text, max_text_len, max_seq_len):
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


def build_bert_classifier_model(bert_model_dir, max_seq_len):
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

    model.compile(
        optimizer=keras.optimizers.Adam(),
        loss=keras.losses.BinaryCrossentropy(),
        metrics=[keras.metrics.BinaryCrossentropy(name="acc")],
    )

    return model
