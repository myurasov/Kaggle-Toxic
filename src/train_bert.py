#!/usr/bin/python

from lib.bert_utils import build_bert_classifier_model

from src.config import config as c

model = build_bert_classifier_model(
    bert_model_dir=c["DATA_DIR"] + "/bert", max_seq_len=c["MAX_SEQ_LENGTH"]
)

model.summary()
