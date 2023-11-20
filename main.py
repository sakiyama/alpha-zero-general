from Coach import Coach
from tictactoe.TicTacToeGame import TicTacToeGame as Game
from tictactoe.network import Network

if __name__ == "__main__":
    game = Game()
    network = Network(game)
    coach = Coach(game, network)
    coach.learn()
