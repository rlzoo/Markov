#!/bin/env/python
# -*- encoding: utf-8 -*-
"""
GridWorld Environment
"""
from __future__ import division, print_function

from markov import MDPEnv, MDPState


class GridWorld(MDPEnv):

    def __init__(self, shape: tuple = (4, 4), ends: list = [(0, 0), (3, 3)]):
        """

        :param shape:
        :param ends:
        """
        super().__init__()
        self.shape = shape
        self.ends = ends

        self.action_space = [self.north, self.south, self.east, self.west]

        non_terminal_value = -1
        terminal_value = 0

        self.grid = [[None for _ in range(shape[1])] for __ in range(shape[0])]

        self.states = []
        for h in range(shape[0]):
            for w in range(shape[1]):
                crd = (h, w)
                actions = []

                terminal = True if crd in ends else False
                reward = terminal_value if terminal else non_terminal_value

                if not terminal:
                    actions = self.action_space

                state = MDPState(reward, actions, terminal, self,
                                 action_args=crd)
                self.grid[h][w] = state
                self.states.append(state)

        for s in self.states:
            s.init_state()

        self.load_states(self.states)

    def north(self, env, crd):
        """

        :param env:
        :param crd:
        :return:
        """
        if crd[0] > 0:
            return env.grid[crd[0] - 1][crd[1]]
        else:
            return env.grid[crd[0]][crd[1]]

    def south(self, env, crd):
        """

        :param env:
        :param crd:
        :return:
        """
        if crd[0] < self.shape[0] - 1:
            return env.grid[crd[0] + 1][crd[1]]
        else:
            return env.grid[crd[0]][crd[1]]

    def east(self, env, crd):
        """

        :param env:
        :param crd:
        :return:
        """
        if crd[1] < self.shape[1] - 1:
            return env.grid[crd[0]][crd[1] + 1]
        else:
            return env.grid[crd[0]][crd[1]]

    def west(self, env, crd):
        """

        :param env:
        :param crd:
        :return:
        """
        if crd[1] > 0:
            return env.grid[crd[0]][crd[1] - 1]
        else:
            return env.grid[crd[0]][crd[1]]

    def print(self):
        """
        Print GridWorld
        """
        for h in range(self.shape[0]):
            print('+---------' * self.shape[1] + '+')
            row = ''
            for w in range(self.shape[1]):
                row += '|   {}'.format(str(round(float(
                    self.grid[h][w].value), 2)).ljust(6))
            print(row + '|')
        print('+---------' * self.shape[1] + '+')