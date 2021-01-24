#!/usr/bin/python

import unittest

import numpy as np
import pandas as pd
from bert.tokenization.bert_tokenization import FullTokenizer
from src.config import config
from lib.bert_utils import preprocess_text_for_bert

SKIP_LONG = False
SKIP_LONG_MESSAGE = "Skipping long tests"

# number of feature coluumns
N_FEATURES = 6


class Validate_Bert_DATA(unittest.TestCase):
    def setUp(self):

        # training data

        self.train_df = pd.read_csv(config["DATA_DIR"] + "/src/train.csv").set_index(
            "id"
        )

        self.train_X = np.load(config["DATA_DIR"] + "/processsed_for_bert/train.X.npy")
        self.train_Y = np.load(config["DATA_DIR"] + "/processsed_for_bert/train.Y.npy")

        # test data

        self.test_df = pd.read_csv(config["DATA_DIR"] + "/src/test.csv").set_index("id")

        self.test_X = np.load(config["DATA_DIR"] + "/processsed_for_bert/test.X.npy")
        self.test_ids = np.load(
            config["DATA_DIR"] + "/processsed_for_bert/test.ids.npy"
        )

    def test_bert_validate_train_types(self):
        dataset_len = len(self.train_df)

        self.assertEqual(self.train_X.dtype, np.int32)
        self.assertEqual(self.train_X.shape, (dataset_len, config["MAX_SEQ_LENGTH"]))

        self.assertEqual(self.train_Y.dtype, np.uint8)
        self.assertEqual(self.train_Y.shape, (dataset_len, N_FEATURES))

    def test_bert_validate_test_types(self):
        dataset_len = len(self.test_df)

        self.assertEqual(self.test_X.dtype, np.int32)
        self.assertEqual(self.test_X.shape, (dataset_len, config["MAX_SEQ_LENGTH"]))

        self.assertEqual(self.test_ids.shape[0], dataset_len)

    @unittest.skipIf(SKIP_LONG, SKIP_LONG_MESSAGE)
    def test_bert_validate_train_Y(self):
        # validate if every n-th row in prepared data matches to the dataframe

        step = 353

        for i in range(0, len(self.train_df), step):
            row = self.train_df.values[i]
            self.assertListEqual(list(row[1:]), list(self.train_Y[i]))

    @unittest.skipIf(SKIP_LONG, SKIP_LONG_MESSAGE)
    def test_bert_validate_train_X(self):
        # validate if every n-th row in prepared data matches to the dataframe

        tokenizer = FullTokenizer(vocab_file=config["DATA_DIR"] + "/bert/vocab.txt")
        step = 353

        # train set preprocessed values
        for i in range(0, len(self.train_df), step):
            row = self.train_df.values[i]

            comment_text = row[0]
            pp = preprocess_text_for_bert(
                tokenizer=tokenizer,
                text=comment_text,
                max_text_len=config["MAX_TEXT_LENGTH"],
                max_seq_len=config["MAX_SEQ_LENGTH"],
            )

            self.assertListEqual(list(pp), list(self.train_X[i]))

        # test set preprocessed values
        for i in range(0, len(self.test_df), step):
            row = self.test_df.values[i]

            comment_text = row[0]
            pp = preprocess_text_for_bert(
                tokenizer=tokenizer,
                text=comment_text,
                max_text_len=config["MAX_TEXT_LENGTH"],
                max_seq_len=config["MAX_SEQ_LENGTH"],
            )

            self.assertListEqual(list(pp), list(self.test_X[i]))


if __name__ == "__main__":
    unittest.main()
