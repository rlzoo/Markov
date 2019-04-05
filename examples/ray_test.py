import numpy as np
import ray

from markov_rlzoo import GreedyPolicy
from markov_rlzoo.envs.gridworld import GridWorld

ray.init()

class Number():

    def __init__(self):
        self.value = 0

a = ray.put(Number())

@ray.remote
def increase(a):
    a.value += 1
    print(a.value)

print(ray.get(increase.remote(a)))
print(ray.get(increase.remote(a)))
print(ray.get(increase.remote(a)))
print(ray.get(increase.remote(a)))
print(ray.get(increase.remote(a)))