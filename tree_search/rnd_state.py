from __future__ import annotations
import sys
import os
import copy
import pickle
import pyspiel
import hashlib
import numpy as np

from typing import TYPE_CHECKING

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from util.rnd_definitions import RewardCodes
from util.rnd_util import pack_flat_map_str
if TYPE_CHECKING:
    from typing import Tuple


class RNDTreeState:

    def __init__(self, map_str: str, reward_structure: int = 0, obs_show_ids: bool = True, same_obs_equal=True):
        env_configs = {"grid": map_str, "obs_show_ids": True, "reward_structure": reward_structure}
        game = pyspiel.load_game("stones_and_gems", env_configs)
        self._state = game.new_initial_state()
        self._show_ids = obs_show_ids
        self._same_obs_equal = same_obs_equal
        self._observation_shape = game.observation_tensor_shape()
        self._state_tensor = self._timestep_to_state()
        self._sample_external_events()

    def _sample_external_events(self):
        while self._state.is_chance_node():
            if self._state.is_chance_node():
                self._state.apply_action(0)

    def _timestep_to_state(self):
        state = np.array(self._state.observation_tensor(0), dtype="uint16").reshape(self._observation_shape)
        if not self._show_ids:
            state = np.array(state > 0, dtype="uint8")
        return state

    def _get_reward_code(self):
        return int(self._state.rewards()[0])

    def successors(self):
        reward_code = self._get_reward_code()
        # Agent dies -> deadlock but not solution
        if reward_code & RewardCodes.kRewardAgentDies > 0:
            return []
        return [i for i in range(RNDTreeState.num_actions())]
    
    @staticmethod
    def num_actions() -> int:
        return 5

    def observation_shape(self) -> Tuple[int, int, int]:
        return self._observation_shape
    
    def is_solution(self):
        reward_code = self._get_reward_code()
        # Solution only if terminal + reason for terminal is walking through exit
        return self._state.is_terminal() and (reward_code & RewardCodes.kRewardWalkThroughExit > 0)
    
    def apply_action(self, action):
        self._state.apply_action(action)
        self._sample_external_events()
        self._state_tensor = self._timestep_to_state()
    
    def get_image_representation(self):
        return self._timestep_to_state()
    
    def heuristic_value(self):
        pass
    
    def reset(self):
        pass

    def __hash__(self):
        if self._same_obs_equal:
            return hash(hashlib.sha1(self._state_tensor).hexdigest())
        else:
            return hash(self._state.serialize())

    def __eq__(self, other):
        if self._same_obs_equal:
            return np.array_equal(self._state_tensor, other._state_tensor)
        else:
            return self._state.serialize() == other._state.serialize()


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
    state1 = RNDTreeState(map_str, 1, same_obs_equal=same_obs_equal)
    state2 = RNDTreeState(map_str, 1, same_obs_equal=same_obs_equal)

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