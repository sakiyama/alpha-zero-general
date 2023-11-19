import Arena
from MCTS import MCTS

from tictactoe.TicTacToeGame import TicTacToeGame
from tictactoe.network import Network
from tictactoe.TicTacToePlayers import RandomPlayer

import numpy as np
from utils import *

if __name__ == '__main__':
    game = TicTacToeGame()
    neural_net = Network

    rp = RandomPlayer(game).play

    config = dotdict({'simulations': 25, 'cpuct': 1.0})
    mcts = MCTS(game, neural_net(game), config)
    n1p = lambda x: np.argmax(mcts.probabilities(x, temp=0))

    arena = Arena.Arena(n1p, rp, game)
    print(arena.playGames(2, verbose=False))
