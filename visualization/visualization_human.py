# -*- coding: utf-8 -*-
"""
Created on Sat Dec  8 13:51:53 2018

@author: initial-h
"""

from __future__ import print_function
from game import Board, Game
from mcts_pure import MCTSPlayer as MCTS_pure
from mcts_alphaZero import MCTSPlayer
from policy_value_net_withGAP_single_pad import PolicyValueNet
import time
from os import path
import os
from collections import defaultdict
import torch

class Human(object):
    """
    human player
    """
    def __init__(self):
        self.player = None

    def set_player_ind(self, p):
        self.player = p

    def get_action(self, board):
        try:
            location = input("Your move: ")
            if isinstance(location, str):  # for python3
                location = [int(n, 10) for n in location.split(",")]
            move = board.location_to_move(location)
        except Exception as e:
            move = -1
        if move == -1 or move not in board.availables:
            print("invalid move")
            move,_ = self.get_action(board)
        return move,None

    def __str__(self):
        return "Human {}".format(self.player)

def run(start_player=0,is_shown=1):
    n = 5
    width, height = 9, 9
    model_file = 'gp_padding3.model'
    p = os.getcwd()
    model_file = path.join(p,model_file)

    board = Board(width=width, height=height, n_in_row=n)
    game = Game(board)


    best_policy = PolicyValueNet(width, height)

    best_policy = PolicyValueNet(board_width=width,board_height=height,model_file=model_file,use_gpu=False)

    alpha_zero_player = MCTSPlayer(best_policy.policy_value_fn, c_puct=5, n_playout=200)

    # play in GUI
    game.start_play_with_UI(alpha_zero_player)


if __name__ == '__main__':
    run(start_player=0,is_shown=True)