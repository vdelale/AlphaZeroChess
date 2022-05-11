import unittest

import Arena
from MCTS import MCTS

from chess_models.ChessGame import ChessGame
from chess_models.ChessPlayers import RandomPlayer
from chess_models.pytorch.NNet import NNetWrapper as ChessPytorchNNet


import numpy as np
from utils import *

class TestChess(unittest.TestCase):

    @staticmethod
    def execute_game_test(game, neural_net):
        rp = RandomPlayer(game).play

        args = dotdict({'numMCTSSims': 25, 'cpuct': 1.0})
        mcts = MCTS(game, neural_net(game), args)
        n1p = lambda x: np.argmax(mcts.getActionProb(x, temp=0))

        arena = Arena.Arena(n1p, rp, game)
        print(arena.playGames(2, verbose=False))

    def test_chess_pytorch(self):
        self.execute_game_test(ChessGame(), ChessPytorchNNet)

if __name__ == '__main__':
    unittest.main()
