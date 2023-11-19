import os
import numpy as np
import sys
from utils import *

from .TicTacToeNNet import TicTacToeNNet as onnet
config = dotdict({
    'lr': 0.001,
    'dropout': 0.3,
    'epochs': 10,
    'batch_size': 64,
    'channels': 512,
})

class NNetWrapper:
    def __init__(self, game):
        self.network = onnet(game, config)

    def train(self, examples):
        input_boards, target_pis, target_vs = list(zip(*examples))
        input_boards = np.asarray(input_boards)
        target_pis = np.asarray(target_pis)
        target_vs = np.asarray(target_vs)
        self.network.model.fit(
            x = input_boards,
            y = [target_pis, target_vs],
            batch_size = config.batch_size,
            epochs = config.epochs
        )

    def predict(self, board):
        board = board[np.newaxis, :, :]
        pi, v = self.network.model.predict(board, verbose=False)
        return pi[0], v[0]

    def save(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        filename = filename.split(".")[0] + ".h5"
        filepath = os.path.join(folder, filename)
        if not os.path.exists(folder):
            os.mkdir(folder)

        self.network.model.save_weights(filepath)

    def load(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        filename = filename.split(".")[0] + ".h5"

        filepath = os.path.join(folder, filename)
        if not os.path.exists(filepath):
            raise("No model in path '{}'".format(filepath))

        self.network.model.load_weights(filepath)
