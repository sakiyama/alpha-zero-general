import math
import numpy as np

class MCTS():
    def __init__(self, network):
        self.network = network
        self.Qsa = {}  # stores Q values for s,a (as defined in the paper)
        self.Nsa = {}  # stores #times edge s,a was visited
        self.visited = {}  # stores #times board s was visited
        self.policy = {}  # stores initial policy (returned by neural net)

    def NsaSelect(self,game,simulations=50) :
        counts = self.NsaCounts(game,simulations)
        counts_sum = float(sum(counts))
        probabilities = [x / counts_sum for x in counts]
        action = np.random.choice(len(probabilities), p=probabilities)
        return action , probabilities

    def bestAction(self,game,simulations=50) :
        counts = self.NsaCounts(game,simulations)
        actions = np.argwhere(counts == np.max(counts)).flatten()
        return np.random.choice(actions)

    def NsaCounts(self,game,simulations) :
        for i in range(simulations):
            self.search(game)

        s = game.id
        Nsa = self.Nsa
        return [
            Nsa[(s, a)]
            if (s, a) in Nsa
            else 0
            for a in range(game.size)
        ]

    def expand(self,game):
        s = game.id

        p, v = self.network.predict(game.encode())
        actionNumbers = game.actionNumbers()
        for index, value in enumerate(p):
            if index not in actionNumbers :
                p[index] = 0
        sum_policy = np.sum(p)

        if sum_policy == 0:
            sum_policy = len(actionNumbers)
            p = [
                1 / sum_policy
                if index in actionNumbers
                else 0
                for index, _ in enumerate(p)
            ]
        else:
            p /= sum_policy
        self.policy[s] = p
        self.visited[s] = 0
        return v[0]

    def puct(self, s, a, cpuct = 1) :
        Qsa = self.Qsa
        Nsa = self.Nsa
        policy = self.policy
        visited = self.visited

        if (s, a) not in Qsa:
            return 0
        return Qsa[(s, a)] + cpuct * policy[s][a] * math.sqrt(visited[s]) / (1 + Nsa[(s, a)])

    def puctSelect(self,game):
        policy = self.policy
        Qsa = self.Qsa
        Nsa = self.Nsa

        s = game.id

        actions = game.actionNumbers()
        pucts = np.array([self.puct(s, a) for a in actions])
        a = actions[np.argmax(pucts)]

        clone = game.clone()
        clone.actionFromNumber(a)
        v = -self.search(clone)

        if (s, a) in Qsa:
            Qsa[(s, a)] = (Nsa[(s, a)] * Qsa[(s, a)] + v) / (Nsa[(s, a)] + 1)
            Nsa[(s, a)] += 1
        else:
            Qsa[(s, a)] = v
            Nsa[(s, a)] = 1

        self.visited[s] += 1
        return v

    def search(self, game):
        if game.done :
            return game.reward()

        if game.id not in self.policy:
            return self.expand(game)

        return self.puctSelect(game)
