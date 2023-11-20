from tictactoe import TicTacToe as Game
from network import Network
from random import shuffle
import numpy as np
from tqdm import tqdm
from Arena import Arena
from MCTS import MCTS

game = Game()
network = Network()
mcts = MCTS(network)

def selfPlay():
    examples = []
    step = 0
    game = Game()

    while game.done == False:
        step += 1
        probabilities = mcts.probabilities(game, step < 15)
        examples.append([game.clone(), probabilities])
        action = np.random.choice(len(probabilities), p=probabilities)
        game.actionFromNumber(action)

    reward = game.reward()
    return [
        (game2.board,probabilities,reward * ((-1) ** (game2.player != game.player)))
        for game2,probabilities in examples
    ]

def test(b4,after,count=20, winRate=0.6) :
    game = Game()
    players = [MCTS(after),MCTS(b4)]

    afterWon = 0
    draw = 0

    for now in range(count):
        choice = np.random.choice([0, 1])
        start = choice
        while not game.done:
            player = players[choice]
            probabilities = player.probabilities(game, False)
            action = np.argmax(probabilities)
            game.actionFromNumber(action)
            choice = (choice + 1) % 2
        # game.show()
        if 0 == game.winner :
            draw += 1
        elif 1 == game.winner and start == 0 :
            afterWon += 1
        elif -1 == game.winner and start == 1 :
            afterWon += 1

    rate = afterWon/count
    print("learned model win rate",rate,"draw",draw/count)
    return rate >= winRate

selfPlayAndLearn = 30
selfPlayGames = 3

# selfPlayAndLearn = 1000,
# selfPlayGames = 100,

compare = 20
winRate = 0.6

for i in range(selfPlayAndLearn):
    examples = []
    for _ in tqdm(range(selfPlayGames), desc="Self Play"):
        mcts = MCTS(network)
        examples.extend(selfPlay())
    b4trainNetwork = network.clone()
    shuffle(examples)
    network.train(examples)

    if test(b4trainNetwork,network,compare) :
        network.save(filename="checkpoint_" + str(i) + ".h5")
        network.save(filename='best.h5')
    else :
        network = b4trainNetwork
