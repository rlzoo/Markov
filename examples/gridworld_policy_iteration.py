#!/bin/env/python
# -*- encoding: utf-8 -*-
"""

"""
from __future__ import division, print_function

import argparse

import numpy as np
import ray
import tqdm

from markov_rlzoo import GreedyPolicy
from markov_rlzoo.envs.gridworld import GridWorld


def sync_value_iteration(K=1, vis_type=0, discount_factor=1.):
    """

    :param K:
    :param discount_factor:
    :return:
    """
    env = GridWorld(shape=(20, 20), ends=[(0, 0), (19, 19)])

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

    if vis_type == 0:
        env.print()
    elif vis_type == 1:
        env.cv2_visualize()
    else:
        raise UserWarning("Invalid vis_type arg: {}".format(vis_type))


@ray.remote
class ValueParameterServer(object):

    def __init__(self):
        self.values = None

    def pull(self):
        if isinstance(self.values, list):
            return self.values
        else:
            return None

    def push(self, values):
        self.values = values


@ray.remote
def worker_task(state_id, ps, discount_factor=1.0):

    # Create the environment within the remote function
    # instead of having to serialize the entire environment
    env = GridWorld(shape=(100, 100), ends=[(0, 0), (3, 3)])
    GreedyPolicy(env)

    values = ray.get(ps.pull.remote())
    if values is not None:
        for i, state in enumerate(env.states):
            env.states[i].value = values[i]

    state = env.states[state_id]

    v = 0

    for i, action in enumerate(state.actions):
        policy = state.policy[i]

        next_state = action(env, state.action_args)
        r = next_state.reward
        v += policy * (r + discount_factor * next_state.value)

    return v


def async_value_iteration(k=1, discount_factor=1., vis_type=0):
    """

    :param K:
    :param discount_factor:
    :return:
    """
    # Environment shape
    shape = (4, 4)

    # Parameter server to share the values of the states
    ps = ValueParameterServer.remote()

    # Perform the value iterations for each state on
    # a remote function instead of serially
    for _ in tqdm.tqdm(range(k), desc="Processing k iterations"):
        values = ray.get([worker_task.remote(state_id, ps) for state_id in
                             range(shape[0] * shape[1])])
        ps.push.remote(values)

    # Visualize the results
    env = GridWorld(shape=shape, ends=[(0, 0), (3, 3)])

    for i, state in enumerate(env.states):
        env.states[i].value = values[i]

    env.print()
    env.cv2_visualize()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", help="number of k-iterations",
                        type=int, default=1)
    parser.add_argument("--dist",
                        help="whether to distribute (0: sync, 1: async)",
                        type=int, default=1)
    parser.add_argument("--vis",
                        help="Visualization type (0: ascii, 1: cv2)",
                        type=int, default=0)
    args = parser.parse_args()
    k = args.k
    dist = args.dist
    vis_type = args.vis

    if dist == 0:
        sync_value_iteration(k, vis_type=vis_type)

    elif dist == 1:
        ray.init()
        async_value_iteration(k, vis_type=vis_type)

    else:
        raise UserWarning("Invalid dist arg: {}".format(dist))


if __name__ == "__main__":
    main()
