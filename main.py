from tictactoe.TicTacToeGame import TicTacToeGame as Game
from tictactoe.network import Network
from random import shuffle
import numpy as np
from tqdm import tqdm
from Arena import Arena
from MCTS import MCTS

game = Game()
network = Network(game)
mcts = MCTS(game, network)

def selfPlay(tempThreshold = 15):
    examples = []
    player = 1
    step = 0
    board = game.board()

    while True:
        step += 1
        temp = int(step < tempThreshold)
        canonicalBoard = game.getCanonicalForm(board, player)
        probabilities = mcts.probabilities(canonicalBoard, temp)
        rotated = game.rotate(canonicalBoard, probabilities)
        for board2, probabilities2 in rotated:
            examples.append([board2, player, probabilities2])

        action = np.random.choice(len(probabilities), p=probabilities)
        board, player = game.next(board, player, action)
        r = game.done(board, player)

        if r != 0:
            return [
                (x[0], x[2], r * ((-1) ** (x[1] != player)))
                for x in examples
            ]



selfPlayAndLearn = 30
selfPlayGames = 3

# selfPlayAndLearn = 1000,
# selfPlayGames = 100,

compare = 40
winRate = 0.6

for i in range(selfPlayAndLearn):
    examples = []
    for _ in tqdm(range(selfPlayGames), desc="Self Play"):
        mcts = MCTS(game, network)
        examples.extend(selfPlay())

    shuffle(examples)
    b4trainNetwork = network.clone()

    pmcts = MCTS(game, b4trainNetwork)

    network.train(examples)
    nmcts = MCTS(game, network)

    arena = Arena(
        lambda x: np.argmax(pmcts.probabilities(x, 0)),
        lambda x: np.argmax(nmcts.probabilities(x, 0)),
        game
    )
    pwins, nwins, draws = arena.playGames(compare)

    if pwins + nwins == 0 or float(nwins) / (pwins + nwins) < winRate:
        network = b4trainNetwork
    else:
        network.save(filename="checkpoint_" + str(i) + ".h5")
        network.save(filename='best.h5')
