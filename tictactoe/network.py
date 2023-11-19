import os
import numpy as np
from utils import *
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *

lerning_rate = 0.001
dropout = 0.3
channels = 512
batch_size = 64
epochs = 10

class Network:
    def __init__(self, game):
        x, y = game.boardSize()
        input_boards = Input(shape=(x, y))    # s: batch_size x board_x x board_y

        x_image = Reshape((x, y, 1))(input_boards)        # batch_size  x board_x x board_y x 1
        h_conv1 = Activation('relu')(BatchNormalization(axis=3)(Conv2D(channels, 3, padding='same')(x_image))) # batch_size  x board_x x board_y x channels
        h_conv2 = Activation('relu')(BatchNormalization(axis=3)(Conv2D(channels, 3, padding='same')(h_conv1))) # batch_size  x board_x x board_y x channels
        h_conv3 = Activation('relu')(BatchNormalization(axis=3)(Conv2D(channels, 3, padding='same')(h_conv2))) # batch_size  x (board_x) x (board_y) x channels
        h_conv4 = Activation('relu')(BatchNormalization(axis=3)(Conv2D(channels, 3, padding='valid')(h_conv3))) # batch_size  x (board_x-2) x (board_y-2) x channels
        h_conv4_flat = Flatten()(h_conv4)
        s_fc1 = Dropout(dropout)(Activation('relu')(BatchNormalization(axis=1)(Dense(1024)(h_conv4_flat)))) # batch_size x 1024
        s_fc2 = Dropout(dropout)(Activation('relu')(BatchNormalization(axis=1)(Dense(512)(s_fc1))))  # batch_size x 1024
        pi = Dense(game.actionSize(), activation='softmax', name='pi')(s_fc2) # batch_size x self.action_size
        v = Dense(1, activation='tanh', name='v')(s_fc2) # batch_size x 1

        self.model = Model(inputs=input_boards, outputs=[pi, v])
        self.model.compile(
            loss=['categorical_crossentropy','mean_squared_error'],
            optimizer=Adam(lerning_rate)
        )

    def train(self, examples):
        input_boards, target_pis, target_vs = list(zip(*examples))
        input_boards = np.asarray(input_boards)
        target_pis = np.asarray(target_pis)
        target_vs = np.asarray(target_vs)
        self.model.fit(
            x = input_boards,
            y = [target_pis, target_vs],
            batch_size = batch_size,
            epochs = epochs
        )

    def predict(self, board):
        board = board[np.newaxis, :, :]
        pi, v = self.model.predict(board, verbose=False)
        return pi[0], v[0]

    def save(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        filename = filename.split(".")[0] + ".h5"
        filepath = os.path.join(folder, filename)
        if not os.path.exists(folder):
            os.mkdir(folder)

        self.model.save_weights(filepath)

    def load(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        filename = filename.split(".")[0] + ".h5"

        filepath = os.path.join(folder, filename)
        if not os.path.exists(filepath):
            raise("No model in path '{}'".format(filepath))

        self.model.load_weights(filepath)
