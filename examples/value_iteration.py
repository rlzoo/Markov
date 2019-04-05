#!/bin/env/python
# -*- encoding: utf-8 -*-
"""

"""
from __future__ import division, print_function
from markov_rlzoo import GreedyPolicy
from markov_rlzoo.envs.gridworld import GridWorld
import numpy as np
import random
import time
import tqdm
import ray


ENV_SIZE = (100, 100)
ENDS = [(0,0), (99, 99)]

def create_env(size=(1000, 1000)):
    pass


def sync_value_iteration(K=1, vis_type=0, discount_factor=1.,
                         shape=ENV_SIZE, ends=ENDS):
    """
    Synchronous Value Iteration

    :param K: Number of iterations
    :param discount_factor: The discount factor for the return
    :return: None
    """
    # Initialize the GridWorld environment
    env = GridWorld(shape=shape, ends=ends)

    # Initialize with a uniform policy
    P = GreedyPolicy(env)

    # Placeholder for state values
    values = np.zeros(env.n_states)

    for k in tqdm.tqdm(range(K), desc="K iterations:"):
        for state in env.states:
            v = 0
            for i, action in enumerate(state.actions):
                policy = state.policy[i]
                next_state = action(env, state.action_args)
                r = -1
                v += policy * (r + discount_factor * next_state.value)

            values[state.index] = v

        for state in env.states:
            state.value = values[state.index]

    # Visualize
    if vis_type == 0:
        env.print()
    elif vis_type == 1:
        env.cv2_visualize()
    elif vis_type == -1:
        pass
    else:
        raise UserWarning("Invalid vis_type arg: {}".format(vis_type))

    return env


@ray.remote
class ValueParameterServer(object):

    def __init__(self, n_states, env_servers):
        # Placeholder for values
        self.env_servers = env_servers
        self.values = [0 for _ in range(n_states)]
        self.worker_status = [0 for _ in range(n_states)]
        self.k = 0

    def send_complete(self, worker_index):
        self.worker_status[worker_index] = 1

    def is_complete(self, n_workers):
        # print(self.worker_status)
        if self.worker_status.count(1) == n_workers:
            return True
        else:
            return False

    def pull(self):
        # Pull the values from the server
        if isinstance(self.values, list):
            return self.values
        else:
            return None

    def fetch_k(self):
        return self.k

    def state_push(self, value, state_index):
        self.values[state_index] = value
        for env in self.env_servers:
            env.update_state_value.remote(value, state_index)

    def push(self, values):
        # Set new values
        self.values = values


@ray.remote
class EnvParameterServer(object):

    def __init__(self):
        # Initialize a GridWorld environment
        self.env = GridWorld(shape=ENV_SIZE, ends=ENDS)
        GreedyPolicy(self.env)

    def update_values(self, values):
        # Update the values in each state
        if values is not None:
            for i, state in enumerate(self.env.states):
                self.env.states[i].value = values[i]

    def update_state_value(self, value, state_index):
        self.env.states[state_index].value = value

    def get_state_data(self, state_index):
        # Get the state data from the state index
        state = self.env.states[state_index]
        actions = state.actions
        action_indexes = [self.env.action_space.index(a) for a in actions]
        policies = state.policy
        action_args = state.action_args
        # print(action_indexes, policies, action_args)
        return action_indexes, policies, action_args

    def eval_action(self, action_index, action_args):
        # Evaluate an action
        action = self.env.action_space[action_index]
        next_state = action(self.env, action_args)
        reward = next_state.reward
        value = next_state.value
        return reward, value


@ray.remote
def worker_task_v3(k_iterations, n_states, value_ps, env_ps, discount_factor=1.0):
    # Notice that the only inputs to the function is a number
    # and two object IDs. Try to minimalize the number of items
    # that have to be serialized and sent to every worker.

    # Get the actions, policies, and required action arguments for
    # a particular state

    # print("GOT HERE!!!")

    for k in range(k_iterations):

        state_id = random.randint(0, n_states - 1)

        # print("Got to checkpoint 1")

        actions, policies, action_args = ray.get(env_ps.get_state_data.remote(state_id))
        v = 0

        # print("Got to checkpoint 2")

        for i, action in enumerate(actions):
            r, next_state_value = ray.get(env_ps.eval_action.remote(action, action_args))
            # print("Got to checkpoint 3")
            # Instead of measuring the reward in terms of the state's return,
            # in GridWorld, the reward is always -1 for any transition
            r = -1

            policy = policies[i]
            v += policy * (r + discount_factor * next_state_value)

        value_ps.state_push.remote(v, state_id)
        # print("Got to checkpoint 4")

    value_ps.send_complete.remote(state_id)


def async_value_iteration_v4(k=1, discount_factor=1., vis_type=0):
    # Environment shape
    shape = ENV_SIZE

    n_workers = 16
    n_states = shape[0] * shape[1]

    # Number of environment servers
    n_envs = 4
    env_servers = [EnvParameterServer.remote() for _ in range(n_envs)]

    # Parameter server to share the values of the states
    value_ps = ValueParameterServer.remote(n_states, env_servers)

    for w in range(n_workers):
        worker_task_v3.remote(k, n_states, value_ps, env_servers[w % n_envs])

    while True:
        complete = ray.get(value_ps.is_complete.remote(n_workers))
        if complete:
            break
        else:
            time.sleep(1)

    values = ray.get(value_ps.pull.remote())

    # Visualize the results
    env = GridWorld(shape=ENV_SIZE, ends=ENDS)

    for i, state in enumerate(env.states):
        env.states[i].value = values[i]

    if vis_type == 0:
        env.print()
    elif vis_type == 1:
        env.cv2_visualize()
    elif vis_type == -1:
        pass
    else:
        raise UserWarning("Invalid vis_type arg: {}".format(vis_type))

    return env


def main():
    ray.init()

    # Compare the method runtimes

    # K iterations
    k = 1000

    start_time = time.time()
    env = sync_value_iteration(k, vis_type=-1)
    print("Sync time elapsed: {}".format(time.time() - start_time))

    start_time = time.time()
    # env = async_value_iteration_v4(k, vis_type=-1)
    print("Async v4 time elapsed: {}".format(time.time() - start_time))

    env.cv2_visualize()


if __name__ == "__main__":
    main()
