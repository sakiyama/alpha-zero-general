from tictactoe import TicTacToe as Game
from network import Network
from random import shuffle
import numpy as np
from tqdm import tqdm
from MCTS import MCTS

def selfPlay(network):
    mcts = MCTS(network)
    examples = []
    game = Game()

    while game.done == False:
        action, probabilities = mcts.NsaSelect(game)
        examples.append([game.clone(), probabilities])
        game.actionFromNumber(action)

    reward = game.reward()
    return [
        (
            game2.board,
            probabilities,
            reward * ((-1) ** (game2.player != game.player))
        )
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
            action = player.bestAction(game)
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

selfPlayAndLearn = 10
selfPlayGames = 2

# selfPlayAndLearn = 1000
# selfPlayGames = 100

network = Network()
for i in range(selfPlayAndLearn):
    examples = []
    for _ in tqdm(range(selfPlayGames), desc="Self Play"):
        examples.extend(selfPlay(network))
    b4trainNetwork = network.clone()
    shuffle(examples)
    network.train(examples)

    if test(b4trainNetwork,network) :
        network.save(filename="checkpoint_" + str(i) + ".h5")
        network.save(filename='best.h5')
    else :
        network = b4trainNetwork
