import sys
from utils import *

from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *

class TicTacToeNNet():
    def __init__(self, game, config):
        x, y = game.boardSize()
        self.config = config
        input_boards = Input(shape=(x, y))    # s: batch_size x board_x x board_y

        x_image = Reshape((x, y, 1))(input_boards)        # batch_size  x board_x x board_y x 1
        h_conv1 = Activation('relu')(BatchNormalization(axis=3)(Conv2D(config.channels, 3, padding='same')(x_image))) # batch_size  x board_x x board_y x channels
        h_conv2 = Activation('relu')(BatchNormalization(axis=3)(Conv2D(config.channels, 3, padding='same')(h_conv1))) # batch_size  x board_x x board_y x channels
        h_conv3 = Activation('relu')(BatchNormalization(axis=3)(Conv2D(config.channels, 3, padding='same')(h_conv2))) # batch_size  x (board_x) x (board_y) x channels
        h_conv4 = Activation('relu')(BatchNormalization(axis=3)(Conv2D(config.channels, 3, padding='valid')(h_conv3))) # batch_size  x (board_x-2) x (board_y-2) x channels
        h_conv4_flat = Flatten()(h_conv4)
        s_fc1 = Dropout(config.dropout)(Activation('relu')(BatchNormalization(axis=1)(Dense(1024)(h_conv4_flat)))) # batch_size x 1024
        s_fc2 = Dropout(config.dropout)(Activation('relu')(BatchNormalization(axis=1)(Dense(512)(s_fc1))))  # batch_size x 1024
        self.pi = Dense(game.actionSize(), activation='softmax', name='pi')(s_fc2) # batch_size x self.action_size
        self.v = Dense(1, activation='tanh', name='v')(s_fc2) # batch_size x 1

        self.model = Model(inputs=input_boards, outputs=[self.pi, self.v])
        self.model.compile(loss=['categorical_crossentropy','mean_squared_error'], optimizer=Adam(config.lr))
