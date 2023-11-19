from Coach import Coach
from tictactoe.TicTacToeGame import TicTacToeGame as Game
from tictactoe.network import Network
from utils import *

config = dotdict({
    'numIters': 1000,
    'episodes': 100,              # Number of complete self-play games to simulate during a new iteration.
    'tempThreshold': 15,        #
    'winRate': 0.6,     # During arena playoff, new neural net will be accepted if threshold or more of games are won.
    'maxlenOfQueue': 200000,    # Number of game examples to train the neural networks.
    'simulations': 25,          # Number of games moves for MCTS to simulate.
    'compare': 40,         # Number of games to play during arena play to determine if new net will be accepted.
    'cpuct': 1,

    'checkpoint': './temp/',
    'load_model': False,
    'load_folder_file': ('/dev/models/8x100x50','best.pth.tar'),
    'maxHistory': 20,
})

if __name__ == "__main__":
    g = Game()
    network = Network(g)
    if config.load_model:
        network.load(config.load_folder_file[0], config.load_folder_file[1])
    c = Coach(g, network, config)
    if config.load_model:
        c.loadExamples()
    c.learn()
