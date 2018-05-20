import numpy as np
import os
import time
import tensorflow as tf

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

    def __init__(self, sess, config=LSTMConfig):
        """
        initialize the LSTM model
        """
        self.sess=sess

        self.input_size=1
        self.num_steps=30
        self.lstm_size=128
        self.num_layers=1

        self.keep_prob=0.8

        self.batch_size=64
        self.init_learning_rate=0.001
        self.learning_rate_decay=0.99
        self.init_epoch=5
        self.max_epoch=50

        # control the size of the embedding used to represent stock
        # symbols
        self.embedding_size=3

        # the number of unique stocks we're learning from
        self.stock_count=10


        self.use_embed = (embed_size is not None) and (embed_size > 0)
        self.embed_size = embed_size or -1

        self.logs_dir="logs"
        self.plots_dir="plot"

    def _create_cell(self):
        lstm_cell = tf.contrib.rnn.LSTMCell(self.lstm_size,
                                            state_is_tuple=True)
        lstm_cell = tf.contrib.rnn.DrooutWrapper(lstm_cell,
                                                 output_keep_prob=self.keep_prob)
        return lstm_cell

    def _stacked_rnn(self):
        if self.num_layers > 1:
            cell = tf.contrib.rnn.MultRNNCell(
                    [self._create_cell() for _ in
                     range(self.num_layers)], state_is_tuple=True
                )
        else:
            cell = self._create_cell()
        return cell

    def build_graphi(self):
        self.learning_rate = tf.placeholder(tf.float32, None,
        name="learning rate")
        self.keep_prob = tf.placeholder(tf.float32, None,
        name="keep_prob")

        self.symbols = None

        cell = self._stacked_rnn()





def init_stacked_lstm():
    if config.num_layers > 1:
        stacked_lstm = tf.contrib.rnn.MultiRNNCell(
            [_create_cell() for _ in range(num_layers)],
            state_is_tuple=True)
        return stacked_lstm
    else:
        return _create_cell()
