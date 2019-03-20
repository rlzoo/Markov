#!/bin/env/python
# -*- encoding: utf-8 -*-
"""

"""
from __future__ import division, print_function

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
        self.env = env
        self.actions = actions
        self.action_args = action_args
        self.value = 0
        self.index = None
        self.prev_states = []
        self.next_states = []

    def init_state(self):
        for a in self.actions:
            next_state = a(self.env, self.action_args)
            if not self in next_state.prev_states:
                next_state.prev_states.append(self)
            self.next_states.append(next_state)


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




def test():
    env = GridWorld()

    env.print()


if __name__ == "__main__":
    test()

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

#
# def policy_eval(policy, env, discount_factor=1.0, theta=0.00001):
#     """
#     Evaluate a policy given an environment and a full description of the
#     environment's dynamics.
#
#     Args:
#         policy: [S, A] shaped matrix representing the policy.
#         env: OpenAI env. env.P represents the transition probabilities of
#         the environment.
#             env.P[s][a] is a list of transition tuples (prob, next_state,
#             reward, done).
#             env.nS is a number of states in the environment.
#             env.nA is a number of actions in the environment.
#         theta: We stop evaluation once our value function change is less
#         than theta for all states.
#         discount_factor: Gamma discount factor.
#
#     Returns:
#         Vector of length env.nS representing the value function.
#     """
#     # Start with a random (all 0) value function
#     V = np.zeros(env.nS)
#     while True:
#         delta = 0
#         # For each state, perform a "full backup"
#         for s in range(env.nS):
#             v = 0
#             # Look at the possible next actions
#             for a, action_prob in enumerate(policy[s]):
#                 # For each action, look at the possible next states...
#                 for prob, next_state, reward, done in env.P[s][a]:
#                     # Calculate the expected value
#                     v += action_prob * prob * (reward + discount_factor *
#                     V[next_state])
#             # How much our value function changed (across any states)
#             delta = max(delta, np.abs(v - V[s]))
#             V[s] = v
#         # Stop evaluating once our value function change is below a threshold
#         if delta < theta:
#             break
#     return np.array(V)
