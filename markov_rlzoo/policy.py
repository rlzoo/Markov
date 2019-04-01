#!/bin/env/python
# -*- encoding: utf-8 -*-
"""

"""
from __future__ import division, print_function


class GreedyPolicy(object):

    def __init__(self,env,init_random=True):
        """

        :param env:
        :param init_random:
        """
        if init_random:
            for state in env.states:
                n_actions = len(env.action_space)
                random_policy = 1. / n_actions

                state.policy = [random_policy for _ in range(n_actions)]


    def update_env(self,env):
        """

        :param env:
        :return:
        """
        for state in env.states:
            rewards = []
            next_states = []
            actions = state.actions

            for a in actions:
                next_state = a(state.env, state.action_args)
                rewards.append(next_state.reward)
                next_states.append(next_state)

            if len(actions) > 0:
                max_reward = max(rewards)
                p = 1. / rewards.count(max_reward)
                a_prob = [p if r == max_reward else 0. for r in rewards]
            else:
                a_prob = []

            state.policy = a_prob


    def evaluate(self, state):
        """

        :param state:
        :return:
        """
        rewards = []
        next_states = []
        actions = state.actions

        for a in actions:
            next_state = a(state.env, state.action_args)
            rewards.append(next_state.reward)
            next_states.append(next_state)

        if len(actions) > 0:
            max_reward = max(rewards)
            p = 1. / rewards.count(max_reward)
            a_prob = [p if r == max_reward else 0. for r in rewards]
        else:
            a_prob = []

        return actions, a_prob, next_states, rewards
