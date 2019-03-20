#!/bin/env/python
# -*- encoding: utf-8 -*-
"""

"""
from __future__ import division, print_function
import numpy as np
import argparse

from markov import GreedyPolicy
from markov.envs.gridworld import GridWorld


def sync_value_iteration(K=1,discount_factor=1.):
    """

    :param K:
    :param discount_factor:
    :return:
    """
    env = GridWorld()

    P = GreedyPolicy(env)

    values = np.zeros(env.n_states)

    for k in range(K):
        for state in env.states:
            v = 0

            for i, action in enumerate(state.actions):
                policy = state.policy[i]
                next_state = action(env, state.action_args)
                r = next_state.reward
                v += policy * (r + discount_factor * next_state.value)

            values[state.index] = v

        for state in env.states:
            state.value = values[state.index]

    env.print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", help="number of k-iterations",
                        type=int,default=1)
    args = parser.parse_args()
    k = args.k

    sync_value_iteration(k)



if __name__ == "__main__":
    main()
