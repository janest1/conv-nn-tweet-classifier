# coding: utf-8

import re
import io
import csv
import mxnet
from mxnet.gluon.data.dataset import ArrayDataset
import gluonnlp as nlp


def load_dataset(train_file, val_file, test_file, max_length=64):
    """
    Inputs: training, validation and test files in TSV format
    Outputs: vocabulary (with attached embedding), training, validation and test datasets ready for neural net training
    """
    train_array = load_tsv_to_array(train_file)
    val_array   = load_tsv_to_array(val_file)
    test_array  = load_tsv_to_array(test_file)
    
    vocabulary  = build_vocabulary(train_array, val_array, test_array)
    train_dataset = preprocess_dataset(train_array, vocabulary, max_length)
    val_dataset = preprocess_dataset(val_array, vocabulary, max_length)
    test_dataset = preprocess_dataset(test_array, vocabulary, max_length)

    return vocabulary, ArrayDataset(train_dataset), ArrayDataset(val_dataset), ArrayDataset(test_dataset)


def load_tsv_to_array(data_file):
    """
    Input: tsv file
    Output: list of tuples (id number, label, text) representing each row in dataset
    """

    tokenizer = nlp.data.SpacyTokenizer()
    array = []
    with open(data_file) as infile:
        for instance in infile.readlines():
            id, label, text = instance.split('\t')
            # tokens = tokenizer(text)
            tokens = ['@' if tok.startswith('@') else tok for tok in tokenizer(text)]
            array.append((id, label, [t.lower() for t in tokens if not t.startswith('http:')]))
            # array.append((id, label, tokens))

    return array


def build_vocabulary(tr_array, val_array, tst_array):
    """
    Inputs: arrays representing the training, validation and test data
    Outputs: vocabulary (Tokenized text as in-place modification of input arrays or returned as new arrays)
    """

    # get all tokens from train, val, and test
    all_tokens = []
    for data_array in [tr_array, val_array, tst_array]:
        for instance in data_array:
            id, label, tokens = instance
            all_tokens.extend(tokens)
    
    counter = nlp.data.count_tokens(all_tokens)
    vocab = nlp.Vocab(counter)

    return vocab


def _preprocess(x, vocab, max_len):
    """
    Inputs: data instance x (tokenized), vocabulary, maximum length of input (in tokens)
    Outputs: data mapped to token IDs, with corresponding label
    """
    id_num, label, text_tokens = x
    # data = vocab[text_tokens.split()]
    data = vocab[text_tokens]

    # restrict or pad instance so it equals max length
    if len(data) > max_len:
        data = data[:64]
    elif len(data) < max_len:
        pad_quant = max_len - len(data)
        for i in range(pad_quant):
            data.append(vocab.token_to_idx['<pad>'])

    if label not in ['Relevant', 'Not Relevant']:
        return None

    if label == 'Relevant':
        label_arr = [1, 0]
    else:
        label_arr = [0, 1]

    return mxnet.nd.array(label_arr), mxnet.nd.array(data)


def preprocess_dataset(dataset, vocab, max_len):
    # filter out None in list (representing a data instance without one of the 2 labels
    preprocessed_dataset = list(filter(None, [_preprocess(x, vocab, max_len) for x in dataset]))
    return preprocessed_dataset


class BasicTransform(object):
    """
    This is a callable object used by the transform method for a dataset. It will be
    called during data loading/iteration.

    Parameters
    ----------
    labels : list string
        List of the valid strings for classification labels
    max_len : int, default 64
        Maximum sequence length - longer seqs will be truncated and shorter ones padded

    """

    def __init__(self, labels, max_len=64):
        self._max_seq_length = max_len
        self._label_map = {}
        for (i, label) in enumerate(labels):
            self._label_map[label] = i

    def __call__(self, label, data):
        label_id = self._label_map[label]
        padded_data = data + [0] * (self._max_seq_length - len(data))
        return mxnet.nd.array(padded_data, dtype='int32'), mxnet.nd.array([label_id], dtype='int32')
