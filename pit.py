import Arena
from MCTS import MCTS
from othello.OthelloGame import OthelloGame
from othello.OthelloPlayers import *
from othello.pytorch.NNet import Network


import numpy as np

"""
use this script to play any two agents against each other, or play manually with
any agent.
"""

human_vs_cpu = True

g = OthelloGame(6)
rp = RandomPlayer(g).play
gp = GreedyOthelloPlayer(g).play
hp = HumanOthelloPlayer(g).play
n1 = Network(g)
n1.load('./pretrained_models/othello/pytorch/','6x100x25_best.pth.tar')
mcts = MCTS(g, n1)
n1p = lambda x: np.argmax(mcts.probabilities(x, 0))
player2 = hp
arena = Arena.Arena(n1p, player2, g, display=OthelloGame.display)

print(arena.playGames(2, verbose=True))
