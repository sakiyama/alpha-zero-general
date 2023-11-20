import Arena
from MCTS import MCTS

from tictactoe.TicTacToeGame import TicTacToeGame
from tictactoe.network import Network
from tictactoe.TicTacToePlayers import RandomPlayer

import numpy as np

if __name__ == '__main__':
    game = TicTacToeGame()
    rp = RandomPlayer(game).play

    mcts = MCTS(game, Network(game))
    n1p = lambda x: np.argmax(mcts.probabilities(x, 0))

    arena = Arena.Arena(n1p, rp, game)
    print(arena.playGames(2))
