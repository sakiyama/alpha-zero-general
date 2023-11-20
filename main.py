from Coach import Coach
from tictactoe.TicTacToeGame import TicTacToeGame as Game
from tictactoe.network import Network
from utils import *

config = dotdict({
    'tempThreshold': 15,        #
    'winRate': 0.6,     # During arena playoff, new neural net will be accepted if threshold or more of games are won.
    'simulations': 25,          # Number of games moves for MCTS to simulate.
    'cpuct': 1,

    'checkpoint': './temp/',
    'maxHistory': 20,
})

if __name__ == "__main__":
    game = Game()
    network = Network(game)
    coach = Coach(game, network, config)
    coach.learn()
