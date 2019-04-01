#!/bin/env/python
# -*- encoding: utf-8 -*-
"""

"""
from __future__ import division, print_function
import numpy as np
import time


class MDPState(object):

    def __init__(self, reward, actions, terminal, env, action_args=None):
        """

        :param reward:
        :param actions:
        :param terminal:
        :param env:
        """
        self.reward = reward
        self.terminal = terminal
        self.env = env
        self.actions = actions
        self.action_args = action_args
        self.value = 0
        self.index = None
        self.prev_states = []
        self.next_states = []
        self.policy = []
        self.global_id = time.time()

    def init_state(self):
        """

        :return:
        """
        for a in self.actions:
            next_state = a(self.env, self.action_args)

            if not self in next_state.prev_states:
                next_state.prev_states.append(self)

            self.next_states.append(next_state)


class MDPEnv(object):

    def load_states(self, states):
        """

        :param states:
        :return:
        """
        self.states = states
        for i, n in enumerate(self.states):
            n.index = i
        self.actions = []
        self.n_states = len(self.states)
        self.n_actions = len(self.actions)

        # State Transition Probability Matrix
        # self.stpm = np.zeros((self.n_states, self.n_states))

