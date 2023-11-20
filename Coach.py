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
    def __init__(self, game, network, config):
        self.game = game
        self.network = network
        self.pnet = self.network.__class__(game)
        self.config = config
        self.mcts = MCTS(game, network, config)
        self.histories = []

    def executeEpisode(self):
        examples = []
        player = 1
        episodeStep = 0
        game = self.game
        board = game.board()

        while True:
            episodeStep += 1
            canonicalBoard = game.getCanonicalForm(board, player)
            temp = int(episodeStep < self.config.tempThreshold)

            probabilities = self.mcts.probabilities(canonicalBoard, temp=temp)
            symmetries = game.symmetries(canonicalBoard, probabilities)
            for b, p in symmetries:
                examples.append([b, player, p, None])

            action = np.random.choice(len(probabilities), p=probabilities)
            board, player = game.next(board, player, action)

            r = game.done(board, player)

            if r != 0:
                return [(x[0], x[2], r * ((-1) ** (x[1] != player))) for x in examples]

    def learn(
        self,
        numIters = 1000,
        episodes = 100,
        maxlenOfQueue = 200000,
        compare = 40,
    ):
        network = self.network
        histories = []
        pnet = self.pnet
        game = self.game

        for i in range(1, numIters + 1):
            history = deque([], maxlen=maxlenOfQueue)
            for _ in tqdm(range(episodes), desc="Self Play"):
                self.mcts = MCTS(game, network, self.config)
                history += self.executeEpisode()
            histories.append(history)

            if len(histories) > self.config.maxHistory:
                histories.pop(0)
            examples = []
            for e in histories:
                examples.extend(e)

            shuffle(examples)
            network.save(filename='temp.h5')
            pnet.load(filename='temp.h5')
            pmcts = MCTS(game, pnet, self.config)

            network.train(examples)
            nmcts = MCTS(game, network, self.config)

            arena = Arena(
                lambda x: np.argmax(pmcts.probabilities(x, temp=0)),
                lambda x: np.argmax(nmcts.probabilities(x, temp=0)),
                game
            )
            pwins, nwins, draws = arena.playGames(compare)

            if pwins + nwins == 0 or float(nwins) / (pwins + nwins) < self.config.winRate:
                network.load(filename='temp.h5')
            else:
                network.save(filename="checkpoint_" + str(iteration) + ".h5")
                network.save(filename='best.h5')
