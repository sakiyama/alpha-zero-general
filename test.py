""""
    This is a Regression Test Suite to automatically test all combinations of games and ML frameworks. Each test
    plays two quick games using an untrained neural network (randomly initialized) against a random player.

    In order for the entire test suite to run successfully, all the required libraries must be installed.  They are:
    Pytorch, Keras.

     [ Games ]      Pytorch      Keras
      -----------   -------      -----
    - TicTacToe                  [Yes]
"""

import Arena
from MCTS import MCTS

from tictactoe.TicTacToeGame import TicTacToeGame
from tictactoe.keras.NNet import NNetWrapper as TicTacToeKerasNNet
from tictactoe.TicTacToePlayers import RandomPlayer

import numpy as np
from utils import *

if __name__ == '__main__':
    game = TicTacToeGame()
    neural_net = TicTacToeKerasNNet

    rp = RandomPlayer(game).play

    config = dotdict({'simulations': 25, 'cpuct': 1.0})
    mcts = MCTS(game, neural_net(game), config)
    n1p = lambda x: np.argmax(mcts.probabilities(x, temp=0))

    arena = Arena.Arena(n1p, rp, game)
    print(arena.playGames(2, verbose=False))
