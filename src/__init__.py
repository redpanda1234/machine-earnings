import os

import numpy as np
import pandas as pd

import tensorflow as tf

from scraper import *
from lstm import *

logging = tf.logging
if not os.path.exists("logs"):
    os.mkdir("logs")

def main():

    hardware_config = tf.ConfigProto()

    with tf.Session(config=hardware_config) as sess:
        lstm = LstmModel(
            sess,


        )

    # Initialize abstract graph representing the flow of computations
    tf.reset_default_graph()
    lstm_graph = tf.Graph()

    inputs = tf.placeholder(tf.float32, [None, config.num_steps, config.input_size])
    targets = tf.placeholder(tf.float32, [None, config.input_size])
    learning_rate = tf.placeholder(tf.float32, None)

    stocks = tf.placeholder(tf.float32, [batch_size, num_steps])

    stacked_lstm = init_stacked_lstm(lstm_size, num_layers, batch_size)

    initial_state = state = stacked_lstm.zero_state(batch_size, tf.float32)
