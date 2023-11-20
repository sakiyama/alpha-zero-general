import os
import sys
from collections import deque
from pickle import Pickler, Unpickler
from random import shuffle
import numpy as np
from tqdm import tqdm

from Arena import Arena
from MCTS import MCTS

class Coach():
    def __init__(self, game, network):
        self.game = game
        self.network = network
        self.mcts = MCTS(game, network)

    def selfPlay(self,tempThreshold = 15):
        examples = []
        player = 1
        step = 0
        game = self.game
        board = game.board()

        while True:
            step += 1
            temp = int(step < tempThreshold)
            canonicalBoard = game.getCanonicalForm(board, player)
            probabilities = self.mcts.probabilities(canonicalBoard, temp)
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

    def learn(
        self,
        selfPlayAndLearn = 30,
        selfPlayGames = 3,

        # selfPlayAndLearn = 1000,
        # selfPlayGames = 100,

        compare = 40,
        winRate = 0.6,
        maxHistory = 20,
    ):
        network = self.network
        game = self.game
        for i in range(selfPlayAndLearn):
            examples = []
            for _ in tqdm(range(selfPlayGames), desc="Self Play"):
                self.mcts = MCTS(game, network)
                examples.extend(self.selfPlay())

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
