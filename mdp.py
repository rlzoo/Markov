#!/bin/env/python
# -*- encoding: utf-8 -*-
"""

"""
from __future__ import print_function, division
import numpy as np


class STPM(object):

    def __init__(self):
        pass


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
        self.actions = actions
        self.action_args = action_args
        self.next_states = [a(env,action_args) for a in actions]
        self.prev_states = []
        self.value = 0
        self.index = None


class MDPEnv(object):

    def load_states(self, states):
        self.states = states
        for i, n in enumerate(self.states):
            n.index = i
        self.actions = []
        self.n_states = len(self.states)
        self.n_actions = len(self.actions)
        # State Transition Probability Matrix
        self.stpm = np.zeros((self.n_states, self.n_states))

    def action_probability(self, state, next_state, action):
        raise self.stpm[state.index][next_state.index]

    def action_reward(self, state, next_state):
        raise next_state.reward


class GridWorld(MDPEnv):

    def __init__(self, shape=(4, 4), ends=[(0, 0), (3, 3)]):
        super().__init__()
        self.shape = shape
        self.ends = ends
        action_space = [
            lambda env, crd: env.grid[crd[0] - 1][crd[1]],
            lambda env, crd: env.grid[crd[0] + 1][crd[1]],
            lambda env, crd: env.grid[crd[0]][crd[1] + 1],
            lambda env, crd: env.grid[crd[0]][crd[1] - 1],
        ]
        action_space_constraints = [
            lambda crd: True if crd[0] > 0 else False,
            lambda crd: True if crd[0] < shape[0] - 1 else False,
            lambda crd: True if crd[1] < shape[1] - 1 else False,
            lambda crd: True if crd[1] > 0 else False
        ]
        non_terminal_value = -1
        terminal_value = 0

        self.grid = [[None for w in range(shape[1])] for h in range(shape[0])]

        states = []
        for h in range(shape[0]):
            for w in range(shape[1]):
                crd = (h,w)
                terminal = True if crd in ends else False
                reward = terminal_value if terminal else non_terminal_value
                actions = []
                if not terminal:
                    for i,rule in enumerate(action_space_constraints):
                        if rule(crd):
                            actions.append(action_space[i])

                states.append(MDPState(reward, actions, terminal, self, action_args=crd))
        self.load_states(states)


    def display(self):
        """
        Display GridWorld
        """
        for h in range(self.shape[0]):
            print('+---------' * self.shape[1] + '+')
            row = ''
            for w in range(self.shape[1]):
                row += '|   {}'.format(str(round(float(
                    self.mdp_space[h][w].value), 2)).ljust(6))
            print(row + '|')
        print('+---------' * self.shape[1] + '+')


    # policy = Policy(mdp)
    # V = Value(mdp, p)
    #
    # for k in range(K):
    #     for state in mdp.states:
    #         v = 0
    #         for next_state in state.next_states:
    #             # Get the probability from the State
    #             # Transition Probability Matrix
    #             P_pss = mdp.sptm(state, next_state, policy)
    #
    # next_s_reward = mdp.reward(state, next_state, policy)
    # next_s_value = V.state_value(next_state)
    #
    #             v += P_pss * (next_s_reward + discount_factor * next_s_value)
    #
    #         s.value = v
    #
    #     for state in mdp.states:
    #         policy.update_policy(state)
    # return policy

    # Taken from Policy Evaluation Exercise!


def policy_eval(policy, env, discount_factor=1.0, theta=0.00001):
    """
    Evaluate a policy given an environment and a full description of the environment's dynamics.

    Args:
        policy: [S, A] shaped matrix representing the policy.
        env: OpenAI env. env.P represents the transition probabilities of the environment.
            env.P[s][a] is a list of transition tuples (prob, next_state, reward, done).
            env.nS is a number of states in the environment.
            env.nA is a number of actions in the environment.
        theta: We stop evaluation once our value function change is less than theta for all states.
        discount_factor: Gamma discount factor.

    Returns:
        Vector of length env.nS representing the value function.
    """
    # Start with a random (all 0) value function
    V = np.zeros(env.nS)
    while True:
        delta = 0
        # For each state, perform a "full backup"
        for s in range(env.nS):
            v = 0
            # Look at the possible next actions
            for a, action_prob in enumerate(policy[s]):
                # For each action, look at the possible next states...
                for prob, next_state, reward, done in env.P[s][a]:
                    # Calculate the expected value
                    v += action_prob * prob * (reward + discount_factor * V[next_state])
            # How much our value function changed (across any states)
            delta = max(delta, np.abs(v - V[s]))
            V[s] = v
        # Stop evaluating once our value function change is below a threshold
        if delta < theta:
            break
    return np.array(V)
