import numpy as np
from random import choice
import copy
import random
x = 3
y = 3
max = 3
size = x * y
class TicTacToe:
    x = x
    y = y
    max = max
    size = size
    def __init__(self):
        self.board = [[0 for _ in range(x)] for _ in range(y)]
        self.player = 1
        self.done = False
        self.id = self._id()

    def encode(self):
        return np.array(self.board)

    def _id(self):
        flattened = [str(item) for sublist in self.board for item in sublist]
        return ''.join(flattened)

    def clone(self) :
        return copy.deepcopy(self)

    @staticmethod
    def action_number(row,col) :
        return row * x + col

    @staticmethod
    def action_row_col(number) :
        return number // x, number % x

    def actions(self) :
        ds = []
        for row in range(y):
            for col in range(x):
                if self.board[row][col] == 0:
                    ds.append([row,col])
        return ds
    def actionNumbers(self) :
        ds = self.actions()
        return [
            TicTacToe.action_number(row,col)
            for row,col in ds
        ]

    def random_action(self) :
        actions = self.actions()
        return random.choice(actions)

    def children(self) :
        children = {}
        for [row,col] in self.actions():
            clone = self.clone()
            clone.action(row,col)
            number = TicTacToe.action_number(row,col)
            children[number] = clone
        return children

    def result(self) :
        board = self.board
        for i in range(max):
            if abs(sum(board[i])) == max:
                return board[i][0]
            if abs(sum(board[j][i] for j in range(max))) == max:
                return board[0][i]
        if abs(sum(board[i][i] for i in range(max))) == max:
            return board[0][0]
        if abs(sum(board[i][2-i] for i in range(max))) == max:
            return board[0][2]
        return 0

    def actionFromNumber(self,number):
        [row,col] = TicTacToe.action_row_col(number)
        self.action(row,col)

    def action(self,row,col):
        board = self.board
        if board[row][col] != 0:
            raise RuntimeError("Invalid action")
        board[row][col] = self.player
        self.id = self._id()

        self.player = -self.player

        winner = self.result()
        space = any(v == 0 for row in board for v in row)

        done = winner != 0 or space is False

        if done :
            self.done = done
            self.winner = winner

    def reward(self) :
        if self.done is False:
            raise RuntimeError(f"game is not done")
        winner = self.winner
        if winner == 0:
            return 0
        elif winner == self.player :
            return 1
        else :
            return -1

    def show(self):
        board = self.board
        to_char = lambda v: ("X" if v == -1 else ("O" if v == 1 else " "))
        rows = [
            [to_char(board[row][col]) for col in range(x)] for row in range(y)
        ]
        cols = ' '.join(str(i) for i in range(x))
        print (
            "\n  " + cols + "\n"
            + "\n".join(str(i) + " " + " ".join(row) for i, row in enumerate(rows))
            + "\n"
        )
