import argparse
import os
import shutil
import time
import random
import numpy as np
import math
import sys
sys.path.append('..')
from utils import *
from NeuralNet import NeuralNet

import argparse
from .TicTacToeNNet import TicTacToeNNet as onnet

"""
NeuralNet wrapper class for the TicTacToeNNet.

Author: Evgeny Tyurin, github.com/evg-tyurin
Date: Jan 5, 2018.

Based on (copy-pasted from) the NNet by SourKream and Surag Nair.
"""

config = dotdict({
    'lr': 0.001,
    'dropout': 0.3,
    'epochs': 10,
    'batch_size': 64,
    'cuda': False,
    'num_channels': 512,
})

class NNetWrapper(NeuralNet):
    def __init__(self, game):
        self.network = onnet(game, config)
        self.board_x, self.board_y = game.boardSize()
        self.action_size = game.actionSize()

    def train(self, examples):
        """
        examples: list of examples, each example is of form (board, pi, v)
        """
        input_boards, target_pis, target_vs = list(zip(*examples))
        input_boards = np.asarray(input_boards)
        target_pis = np.asarray(target_pis)
        target_vs = np.asarray(target_vs)
        self.network.model.fit(x = input_boards, y = [target_pis, target_vs], batch_size = config.batch_size, epochs = config.epochs)

    def predict(self, board):
        """
        board: np array with board
        """
        board = board[np.newaxis, :, :]
        pi, v = self.network.model.predict(board, verbose=False)
        return pi[0], v[0]

    def save(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        # change extension
        filename = filename.split(".")[0] + ".h5"
        filepath = os.path.join(folder, filename)
        if not os.path.exists(folder):
            os.mkdir(folder)

        self.network.model.save_weights(filepath)

    def load(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        # change extension
        filename = filename.split(".")[0] + ".h5"

        # https://github.com/pytorch/examples/blob/master/imagenet/main.py#L98
        filepath = os.path.join(folder, filename)
        if not os.path.exists(filepath):
            raise("No model in path '{}'".format(filepath))

        self.network.model.load_weights(filepath)
