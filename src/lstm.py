import time

import numpy as np
import tensorflow as tf

import util

from tensorflow.python.client import devide_lib

import scraper

logging = tf.logging


class LSTMConfig:

    def __init__()
    input_size=1
    num_steps=30
    lstm_size=128
    num_layers=1
    keep_prob=0.8
    batch_size=64
    init_learining_rate=0.001
    learning_rate_decay=0.99
    init_epoch=5
    max_epoch=50

    # control the size of the embedding used to represent stock
    # symbols
    embedding_size=3

    # the number of unique stocks we're learning from
    stock_count=10


class InputData:
    """
    The object representation of our input data. Lots of this code was
    made by following the structure of tensorflow's LSTM tutorial. As
    we become more experienced with the library, we will change the
    architecture to suit our personal needs. However, for now, we'll
    be following their code pretty closely.
    """

    def __init__(self, config, data, name=None):
        """

        """
        self.batch_size = batch_size = config.batch_size
        self.num_steps = num_steps = config.num_steps
        self.epoch_size = ((len(data) // batch_size) - 1) // num_steps

        # TODO: figure out how to modify scraper to be helpful here
        self.input_data, self.targets = None

class LSTMModel():
    """
    The LSTM model we'll be applying to our stock data.
    """

    def __init__(self, is_training, config, input_):
        """
        initialize the LSTM model
        """
        self._is_training = is_training
        self._input = input_

        # aaa
        self._rnn_params = None
        self._cell = None

        self.batch_size = input_.batch_size
        self.num_steps = input_.num_steps
        size = config.hidden_size



def _create_cell():
    if config.keep_prob < 1.0:
        return tf.contrib.rnn.DropoutWrapper(
            lstm_cell, output_keep_prob=config.keep_prob)
    else:
        return tf.contrib.rnn.LSTMCell(config.lstm_size,
                                       state_is_tuple=True)


def init_stacked_lstm():
    if config.num_layers > 1:
        stacked_lstm = tf.contrib.rnn.MultiRNNCell(
            [_create_cell() for _ in range(num_layers)],
            state_is_tuple=True)
        return stacked_lstm
    else:
        return _create_cell()



def main():

    # Initialize abstract graph representing the flow of computations
    tf.reset_default_graph()
    lstm_graph = tf.Graph()

    inputs = tf.placeholder(tf.float32, [None, config.num_steps, config.input_size])
    targets = tf.placeholder(tf.float32, [None, config.input_size])
    learning_rate = tf.placeholder(tf.float32, None)

    stocks = tf.placeholder(tf.float32, [batch_size, num_steps])

    stacked_lstm = init_stacked_lstm(lstm_size, num_layers, batch_size)

    initial_state = state = stacked_lstm.zero_state(batch_size, tf.float32)

    for i in range(num_steps):
        # value is updated after each processing
        output, state = lstm(stocks[:, i], state)


        # the rest of the code
        # ...



    final_state = state

    # # embedding_matrix is a tensor of shape [vocabulary_size, embedding size]
    # word_embeddings = tf.nn.embedding_lookup(embedding_matrix, word_ids)
