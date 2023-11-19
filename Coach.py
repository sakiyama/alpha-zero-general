import logging
import os
import sys
from collections import deque
from pickle import Pickler, Unpickler
from random import shuffle

import numpy as np
from tqdm import tqdm

from Arena import Arena
from MCTS import MCTS

log = logging.getLogger(__name__)


class Coach():
    def __init__(self, game, network, config):
        self.game = game
        self.network = network
        self.pnet = self.network.__class__(self.game)  # the competitor network
        self.config = config
        self.mcts = MCTS(self.game, self.network, self.config)
        self.histories = []  # history of examples from config.maxHistory latest iterations
        self.skipFirstSelfPlay = False  # can be overriden in loadExamples()

    def executeEpisode(self):
        examples = []
        board = self.game.board()
        self.player = 1
        episodeStep = 0

        while True:
            episodeStep += 1
            canonicalBoard = self.game.getCanonicalForm(board, self.player)
            temp = int(episodeStep < self.config.tempThreshold)

            probabilities = self.mcts.probabilities(canonicalBoard, temp=temp)
            symmetries = self.game.symmetries(canonicalBoard, probabilities)
            for b, p in symmetries:
                examples.append([b, self.player, p, None])

            action = np.random.choice(len(probabilities), p=probabilities)
            board, self.player = self.game.next(board, self.player, action)

            r = self.game.done(board, self.player)

            if r != 0:
                return [(x[0], x[2], r * ((-1) ** (x[1] != self.player))) for x in examples]

    def learn(self):
        for i in range(1, self.config.numIters + 1):
            # bookkeeping
            # examples of the iteration
            if not self.skipFirstSelfPlay or i > 1:
                iterationTrainExamples = deque([], maxlen=self.config.maxlenOfQueue)

                for _ in tqdm(range(self.config.episodes), desc="Self Play"):
                    self.mcts = MCTS(self.game, self.network, self.config)  # reset search tree
                    iterationTrainExamples += self.executeEpisode()

                # save the iteration examples to the history
                self.histories.append(iterationTrainExamples)

            if len(self.histories) > self.config.maxHistory:
                self.histories.pop(0)
            # backup history to a file
            # NB! the examples were collected using the model from the previous iteration, so (i-1)
            self.saveExamples(i - 1)

            # shuffle examples before training
            examples = []
            for e in self.histories:
                examples.extend(e)
            shuffle(examples)

            # training new network, keeping a copy of the old one
            self.network.save(folder=self.config.checkpoint, filename='temp.pth.tar')
            self.pnet.load(folder=self.config.checkpoint, filename='temp.pth.tar')
            pmcts = MCTS(self.game, self.pnet, self.config)

            self.network.train(examples)
            nmcts = MCTS(self.game, self.network, self.config)

            log.info('PITTING AGAINST PREVIOUS VERSION')
            arena = Arena(lambda x: np.argmax(pmcts.probabilities(x, temp=0)),
                          lambda x: np.argmax(nmcts.probabilities(x, temp=0)), self.game)
            pwins, nwins, draws = arena.playGames(self.config.compare)

            log.info('NEW/PREV WINS : %d / %d ; DRAWS : %d' % (nwins, pwins, draws))
            if pwins + nwins == 0 or float(nwins) / (pwins + nwins) < self.config.winRate:
                log.info('REJECTING NEW MODEL')
                self.network.load(folder=self.config.checkpoint, filename='temp.pth.tar')
            else:
                log.info('ACCEPTING NEW MODEL')
                self.network.save(folder=self.config.checkpoint, filename=self.checkpoint(i))
                self.network.save(folder=self.config.checkpoint, filename='best.pth.tar')

    def checkpoint(self, iteration):
        return 'checkpoint_' + str(iteration) + '.pth.tar'

    def saveExamples(self, iteration):
        folder = self.config.checkpoint
        if not os.path.exists(folder):
            os.makedirs(folder)
        filename = os.path.join(folder, self.checkpoint(iteration) + ".examples")
        with open(filename, "wb+") as f:
            Pickler(f).dump(self.histories)

    def loadExamples(self):
        modelFile = os.path.join(self.config.load_folder_file[0], self.config.load_folder_file[1])
        examplesFile = modelFile + ".examples"
        if not os.path.isfile(examplesFile):
            log.warning(f'File "{examplesFile}" with examples not found!')
            r = input("Continue? [y|n]")
            if r != "y":
                sys.exit()
        else:
            log.info("File with examples found. Loading it...")
            with open(examplesFile, "rb") as f:
                self.histories = Unpickler(f).load()
            log.info('Loading done!')

            # examples based on the model were already collected (loaded)
            self.skipFirstSelfPlay = True
