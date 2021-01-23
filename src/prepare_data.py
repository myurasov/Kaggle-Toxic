"""
Convert CSV files to BERT format
Outputs results in .npy format in _data/preprocessed
"""

import os
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd
from bert.tokenization.bert_tokenization import FullTokenizer
from tqdm import tqdm

from src.config import config


# convert piece of text into tokenized version for bert
def preprocess_text_for_bert(tokenizer, text, max_text_len, max_seq_len):

    # tokenize
    tokens = tokenizer.tokenize(text[:max_text_len])
    tokens = tokens[: max_seq_len - 2]
    tokens = ["[CLS]"] + tokens + ["[SEP]"]
    token_ids = tokenizer.convert_tokens_to_ids(tokens)

    # pad
    token_ids += [0] * (max_seq_len - len(token_ids))

    return token_ids


# mapping of comments->tokenized inputs
def _comment_mapping_fn(text):
    return preprocess_text_for_bert(
        tokenizer=tokenizer,
        max_text_len=max_text_len,
        max_seq_len=max_seq_len,
        text=text,
    )


# process csv file
def process_csv(csv_path):

    df = pd.read_csv(csv_path)
    df = df.set_index("id")
    comments = df["comment_text"].to_list()

    # run in parallel
    with Pool(cpu_count()) as pool:

        X = list(
            tqdm(
                pool.imap(
                    _comment_mapping_fn,
                    comments,
                ),
                total=len(df),
            )
        )

    ids = np.array(df.index.to_list())
    X = np.array(X, dtype=np.int32)
    y_df = df[df.columns[2:]]
    Y = np.array(y_df.to_dict("split")["data"], dtype=np.uint8)

    return {"ids": ids, "X": X, "Y": Y}


###

tokenizer = FullTokenizer(vocab_file=config["DATA_DIR"] + "/bert/vocab.txt")
max_text_len = config["MAX_TEXT_LENGTH"]
max_seq_len = config["MAX_SEQ_LENGTH"]

###

os.makedirs(config["DATA_DIR"] + "/processsed", exist_ok=True)

###

print("Processing train data")
processed_data = process_csv(config["DATA_DIR"] + "/src/train.csv")
np.save(config["DATA_DIR"] + "/processsed/train.npy", processed_data)

print("Processing test data")
processed_data = process_csv(config["DATA_DIR"] + "/src/test.csv")
np.save(config["DATA_DIR"] + "/processsed/test.npy", processed_data)
