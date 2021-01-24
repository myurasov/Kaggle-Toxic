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
