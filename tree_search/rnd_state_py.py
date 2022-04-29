from __future__ import annotations
import sys
import os
import copy
import numpy as np

from typing import TYPE_CHECKING

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from rnd_py.rnd_game import RNDGameState
from util.rnd_definitions import RewardCodes
if TYPE_CHECKING:
    from typing import Tuple


class RNDTreeStatePy:

    def __init__(self, map_str: str, obs_show_ids: bool = False, same_obs_equal=True, use_noop=True):
        self._env_configs = {"grid": map_str, "obs_show_ids": obs_show_ids}
        self._state = RNDGameState(self._env_configs)
        self._use_noop = use_noop
        self._show_ids = obs_show_ids
        self._step = 0
        self._same_obs_equal = same_obs_equal
        # self._state_tensor = self.get_image_representation()

    def _get_reward_code(self):
        return self._state.get_reward_signal()

    def successors(self):
        # Agent dies -> deadlock but not solution
        if self._state.is_terminal() and not self._state.is_solution():
            return []
        return [i for i in range(RNDTreeStatePy.num_actions())]
    
    @staticmethod
    def num_actions() -> int:
        return 5

    def observation_shape(self) -> Tuple[int, int, int]:
        return self._state.observation_shape()
    
    def is_solution(self):
        return self._state.is_solution()

    def is_terminal(self):
        return self._state.is_terminal()
    
    def apply_action(self, action):
        self._step += 1
        self._state.apply_action(action)
        # self._state_tensor = self.get_image_representation()
    
    def get_image_representation(self):
        return self._state.get_observation()
    
    def heuristic_value(self) -> int:
        return self._state.heuristic()
    
    def reset(self):
        self._step = 0
        self._state = RNDGameState(self._env_configs)

    def __hash__(self):
        if self._same_obs_equal:
            return hash(self._state)
        else:
            return hash(self._state) + self._step

    def __eq__(self, other):
        # same_state = np.array_equal(self._state_tensor, other._state_tensor)
        same_state = hash(self._state) == hash(other._state)
        if self._same_obs_equal:
            return same_state
        else:
            return same_state and self._step == other._step


def main():
    map_str =   "12|12|999999|1\n" \
                "18|18|18|18|18|18|18|18|18|18|18|18\n" \
                "18|02|02|02|02|02|02|02|02|02|02|18\n" \
                "18|02|02|02|02|02|04|02|02|02|02|18\n" \
                "18|02|02|02|02|02|02|02|02|02|02|18\n" \
                "18|02|02|02|02|02|01|02|02|02|02|18\n" \
                "18|02|02|02|00|02|01|02|02|02|02|18\n" \
                "18|02|02|02|02|02|01|02|02|02|02|18\n" \
                "18|02|02|02|02|02|01|02|02|02|02|18\n" \
                "18|02|02|02|02|02|02|02|02|02|02|18\n" \
                "18|02|02|02|02|02|02|02|02|02|02|18\n" \
                "18|02|02|02|02|02|02|02|02|02|08|18\n" \
                "18|18|18|18|18|18|18|18|18|18|18|18"
    # env_configs = {"grid": map_str, "obs_show_ids": True, "reward_structure": 0}
    # game = pyspiel.load_game("stones_and_gems", env_configs)

    same_obs_equal = True
    state1 = RNDTreeStatePy(map_str, 1, same_obs_equal=same_obs_equal)
    state2 = RNDTreeStatePy(map_str, 1, same_obs_equal=same_obs_equal)

    state2 = copy.deepcopy(state1)

    print(state1 == state2)
    print("hash s1 {}".format(hash(state1)))
    print("hash s2 {}".format(hash(state2)))
    print()

    state1.apply_action(1)
    print(state1 == state2)
    print("hash s1 {}".format(hash(state1)))
    print("hash s2 {}".format(hash(state2)))
    print()

    state2.apply_action(1)
    print(state1 == state2)
    print("hash s1 {}".format(hash(state1)))
    print("hash s2 {}".format(hash(state2)))
    print()

    state2.apply_action(0)
    print(state1 == state2)
    print("hash s1 {}".format(hash(state1)))
    print("hash s2 {}".format(hash(state2)))
    print()

    state2.apply_action(2)
    print(state1 == state2)
    print("hash s1 {}".format(hash(state1)))
    print("hash s2 {}".format(hash(state2)))

    state1 = copy.deepcopy(state2)
    print(state1 == state2)



if __name__ == "__main__":
    main()