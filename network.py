import os
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# 0 = all messages are logged (default behavior)
# 1 = INFO messages are not printed
# 2 = INFO and WARNING messages are not printed
# 3 = INFO, WARNING, and ERROR messages are not printed
from tictactoe import TicTacToe as Game

from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *

lerning_rate = 0.001
dropout = 0.3
channels = 512
batch_size = 64
epochs = 10
folder = './temp/'


class Network:
    def __init__(self, weights = None):
        x = Game.x
        y = Game.y
        input_boards = Input(shape=(x, y))

        x_image = Reshape((x, y, 1))(input_boards)
        h_conv1 = Activation('relu')(BatchNormalization(axis=3)(Conv2D(channels, 3, padding='same')(x_image)))
        h_conv2 = Activation('relu')(BatchNormalization(axis=3)(Conv2D(channels, 3, padding='same')(h_conv1)))
        h_conv3 = Activation('relu')(BatchNormalization(axis=3)(Conv2D(channels, 3, padding='same')(h_conv2)))
        h_conv4 = Activation('relu')(BatchNormalization(axis=3)(Conv2D(channels, 3, padding='valid')(h_conv3)))
        h_conv4_flat = Flatten()(h_conv4)
        s_fc1 = Dropout(dropout)(Activation('relu')(BatchNormalization(axis=1)(Dense(1024)(h_conv4_flat))))
        s_fc2 = Dropout(dropout)(Activation('relu')(BatchNormalization(axis=1)(Dense(512)(s_fc1))))
        pi = Dense(Game.size, activation='softmax', name='pi')(s_fc2)
        v = Dense(1, activation='tanh', name='v')(s_fc2)

        model = Model(inputs=input_boards, outputs=[pi, v])
        model.compile(
            loss=['categorical_crossentropy','mean_squared_error'],
            optimizer=Adam(lerning_rate)
        )
        if weights :
            model.set_weights(weights)
        self.model = model

    def train(self, examples):
        boards, policy, value = list(zip(*examples))
        boards = np.asarray(boards)
        policy = np.asarray(policy)
        value = np.asarray(value)
        self.model.fit(
            x = boards,
            y = [policy, value],
            batch_size = batch_size,
            epochs = epochs
        )

    def predict(self, board):
        board = board[np.newaxis, :, :]
        pi, v = self.model.predict(board, verbose=False)
        return pi[0], v[0]

    def clone(self) :
        weights = self.model.get_weights()
        return Network(weights)

    def save(self, filename):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(folder):
            os.mkdir(folder)

        self.model.save_weights(filepath)

    def load(self, filename):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(filepath):
            raise("No model in path '{}'".format(filepath))

        self.model.load_weights(filepath)
