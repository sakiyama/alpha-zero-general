import math
import numpy as np

EPS = 1e-8

class MCTS():
    def __init__(self, game, network, simulations=50, cpuct=1):
        self.game = game
        self.cpuct = cpuct
        self.simulations = simulations
        self.network = network
        self.Qsa = {}  # stores Q values for s,a (as defined in the paper)
        self.Nsa = {}  # stores #times edge s,a was visited
        self.visited = {}  # stores #times board s was visited
        self.policy = {}  # stores initial policy (returned by neural net)

        self.done = {}  # stores game.done ended for board s
        self.Vs = {}  # stores game.moves for board s

    def probabilities(self, canonicalBoard, temp=1):
        game = self.game
        Nsa = self.Nsa

        for i in range(self.simulations):
            self.search(canonicalBoard)

        s = game.string(canonicalBoard)
        counts = [Nsa[(s, a)] if (s, a) in Nsa else 0 for a in range(game.actionSize())]

        if temp == 0:
            bestAs = np.array(np.argwhere(counts == np.max(counts))).flatten()
            bestA = np.random.choice(bestAs)
            probs = [0] * len(counts)
            probs[bestA] = 1
            return probs

        counts = [x ** (1. / temp) for x in counts]
        counts_sum = float(sum(counts))
        probs = [x / counts_sum for x in counts]
        return probs

    def search(self, canonicalBoard, cpuct = 1):
        game = self.game
        done = self.done
        policy = self.policy
        Qsa = self.Qsa
        Nsa = self.Nsa
        Vs = self.Vs
        visited = self.visited

        s = game.string(canonicalBoard)

        if s not in done:
            done[s] = game.done(canonicalBoard, 1)
        if done[s] != 0:
            return -done[s]

        if s not in policy:
            # leaf node
            policy[s], v = self.network.predict(canonicalBoard)
            valids = game.moves(canonicalBoard, 1)
            policy[s] = policy[s] * valids
            sum_Ps_s = np.sum(policy[s])
            if sum_Ps_s > 0:
                policy[s] /= sum_Ps_s
            else:
                policy[s] = policy[s] + valids
                policy[s] /= np.sum(policy[s])

            Vs[s] = valids
            visited[s] = 0
            return -v

        valids = Vs[s]
        cur_best = -float('inf')
        best_act = -1

        for a in range(game.actionSize()):
            if valids[a]:
                if (s, a) in Qsa:
                    u = Qsa[(s, a)] + cpuct * policy[s][a] * math.sqrt(visited[s]) / (
                            1 + Nsa[(s, a)])
                else:
                    u = cpuct * policy[s][a] * math.sqrt(visited[s] + EPS)
                if u > cur_best:
                    cur_best = u
                    best_act = a

        a = best_act
        next_s, next_player = game.next(canonicalBoard, 1, a)
        next_s = game.getCanonicalForm(next_s, next_player)

        v = self.search(next_s)
        if (s, a) in Qsa:
            Qsa[(s, a)] = (Nsa[(s, a)] * Qsa[(s, a)] + v) / (Nsa[(s, a)] + 1)
            Nsa[(s, a)] += 1
        else:
            Qsa[(s, a)] = v
            Nsa[(s, a)] = 1

        visited[s] += 1
        return -v
