from tqdm import tqdm

class Arena():
    def __init__(self, player1, player2, game):
        self.player1 = player1
        self.player2 = player2
        self.game = game
    def playGame(self):
        game = self.game

        players = [self.player2, None, self.player1]
        player = 1
        board = game.board()
        while game.done(board, player) == 0:
            canonicalForm = game.getCanonicalForm(board, player)
            action = players[player + 1](canonicalForm)
            valids = game.moves(canonicalForm, 1)

            if valids[action] == 0:
                assert valids[action] > 0

            board, player = game.next(board, player, action)
        return player * game.done(board, player)

    def playGames(self, num):
        num = int(num / 2)
        oneWon = 0
        twoWon = 0
        draws = 0
        for _ in tqdm(range(num), desc="Arena.playGames (1)"):
            gameResult = self.playGame()
            if gameResult == 1:
                oneWon += 1
            elif gameResult == -1:
                twoWon += 1
            else:
                draws += 1

        self.player1, self.player2 = self.player2, self.player1

        for _ in tqdm(range(num), desc="Arena.playGames (2)"):
            gameResult = self.playGame()
            if gameResult == -1:
                oneWon += 1
            elif gameResult == 1:
                twoWon += 1
            else:
                draws += 1

        return oneWon, twoWon, draws
