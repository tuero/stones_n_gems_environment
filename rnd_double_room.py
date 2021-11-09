import os 
import sys
import logging
from open_spiel.python import rl_environment

sys.path.append(os.path.dirname(__file__))
from base_rnd import RNDBaseEnv

from util.rnd_util import create_empty_map, create_empty_room
from util.rnd_util import add_item_border_corner, add_item_inside_room, add_room_to_map
from util.rnd_util import get_blocked_idx_corner, get_room_positions_corner, get_room_offset_corner
from util.rnd_definitions import HiddenCellType, hiddencell_to_mapstr, RewardCodes

class RNDDoubleRoom(RNDBaseEnv):
    def __init__(
        self,
        map_size: int = 20,
        max_steps: int = 1000,
        use_noop: bool = True,
        env_mode: int = 2,
        seed: int = 0,
        randomize_on_reset: bool = True,
        render_width: int = -1,
        render_height: int = -1,
        tensor_width: int = 320,
        tensor_height: int = 320,
    ):
        """RND environment for navigating to the open exit.

        Args:
            map_details: map storage object, should have key of "map_id" which holds tile ids of map
            base_dir: Base directory for the map_details if not using single map
            max_steps: Maximum number of steps before environment is over.
            use_noop: Flag to use noop action of standing still
            env_mode: The mode the stones_n_gems environment is using (see implementation)
            seed: Seed for the environment
            randomize_on_reset: Flag to randomize environment on reset
            render_width: Width of the image when rendered
            render_height: Height of the imeage when rendered
            tensor_width: Width of the tensor representation of state image
            tensor_height: Height of the tensor representation of state image
        """
        super().__init__(
            max_steps=max_steps,
            use_noop=use_noop,
            env_mode=env_mode,
            seed=seed,
            randomize_on_reset=randomize_on_reset,
            render_width=render_width,
            render_height=render_height,
            tensor_width=tensor_width,
            tensor_height=tensor_height,
        )
        self._map_size = map_size

    def _create_map(self):
        self._reset_rnd()
        m = create_empty_map(self._map_size)
        room1 = create_empty_room()
        room2 = create_empty_room()
        room_positions = get_room_positions_corner(2)
        room_offset1 = get_room_offset_corner(m, room1, room_positions[0])
        room_offset2 = get_room_offset_corner(m, room2, room_positions[1])

        add_item_inside_room(room1, HiddenCellType.kExitClosed)
        add_item_border_corner(room1, HiddenCellType.kGateRedClosed, room_positions[0])
        add_item_inside_room(room2, HiddenCellType.kKeyRed)
        add_item_border_corner(room2, HiddenCellType.kGateYellowClosed, room_positions[1])

        blocked_idxs1 = get_blocked_idx_corner(m, room1, room_positions[0])
        blocked_idxs2 = get_blocked_idx_corner(m, room2, room_positions[1])
        blocked_idxs = blocked_idxs1 + blocked_idxs2
        add_room_to_map(m, room1, room_offset1)
        add_room_to_map(m, room2, room_offset2)
        
        add_item_inside_room(m, HiddenCellType.kKeyYellow, blocked_tiles=blocked_idxs)
        add_item_inside_room(m, HiddenCellType.kDiamond, blocked_tiles=blocked_idxs)
        add_item_inside_room(m, HiddenCellType.kAgent, blocked_tiles=blocked_idxs)
        return m

    def _get_reward(self):
        reward = int(super()._get_reward())
        if reward & RewardCodes.kRewardAgentDies:
            return -1.0
        elif reward & RewardCodes.kRewardCollectDiamond:
            return 1.0
        elif reward & RewardCodes.kRewardWalkThroughExit:
            return 2.0
        else:
            return 0.0

    def _reset(self):
        self._reset_internal_metrics()
        # Disable the internal logger
        logging.getLogger().setLevel(logging.WARNING)

        # Create map and convert into input string representation
        m = self._create_map()
        
        map_str = hiddencell_to_mapstr(m, self._max_steps, 1)
        env_configs = {"grid": map_str, "obs_show_ids": True, "reward_structure": self._env_mode}
        self._env = rl_environment.Environment("stones_and_gems", **env_configs)

        # Return logging to previous state
        logging.getLogger().setLevel(logging.INFO)
        # Return current time step observation
        return self._timestep_to_state(self._env.reset())
