from ptu.rl.base_environment import BaseEnvironment

import random
import os
import sys
import numpy as np
import torch
import cv2
import logging
from open_spiel.python import rl_environment

sys.path.append(os.path.dirname(__file__))
from util.rnd_definitions import tilestr_to_visiblecellid, hiddencell_to_mapstr
from util.img_draw import rnd_state_to_img


# kRewardDefault = 0,
# kStatic = 1
# kRewardOnlyCompletition= 2,
# kRewardCustom = 3
class RNDBaseEnv(BaseEnvironment):
    def __init__(
        self,
        map_details: np.ndarray = None,
        base_dir: str = None,
        max_steps: int = 1000,
        use_noop: bool = False,
        env_mode: int = 1,
        render_width: int = -1,
        render_height: int = -1,
        tensor_width: int = 320,
        tensor_height: int = 320,
    ):
        """Base RND environment. Can either be created by giving an individual map details,
        or a base directory containing map detail files.

        Args:
            map_details: map storage object, should have key of "map_id" which holds tile ids of map
            base_dir: Base directory for the map_details if not using single map
            max_steps: Maximum number of steps before environment is over.
            use_noop: Flag to use noop action of standing still
            env_mode: The mode the stones_n_gems environment is using (see implementation)
            render_width: Width of the image when rendered
            render_height: Height of the imeage when rendered
            tensor_width: Width of the tensor representation of state image
            tensor_height: Height of the tensor representation of state image
        """
        super().__init__()
        assert map_details is not None or (base_dir is not None and os.path.exists(base_dir))

        # Get args passed in
        self._max_steps = max_steps
        self.map_details = map_details
        self._base_dir = base_dir
        self._num_actions = 5 if use_noop else 4
        self._env_mode = env_mode
        self._render_width = render_width
        self._render_height = render_height
        self._tensor_width = tensor_width
        self._tensor_height = tensor_height

        self._reset_internal_metrics()

    def _reset_internal_metrics(self):
        self._done = False
        self._step = 0
        self._cumulative_reward = 0

    def _timestep_to_state(self, time_step):
        # openspiel internal representation of state are info_states at time_steps.
        base_obs_shape = self._env.game.observation_tensor_shape()
        return np.array(time_step.observations["info_state"][0], dtype="uint8").reshape(base_obs_shape)

    def _get_random_map(self):
        # Choose a random map file from the saved base directory
        assert self._base_dir is not None
        files = len(next(os.walk(self._base_dir))[2])
        assert len(files) > 0
        rand_file = random.choice(files)
        f = os.path.join(self._base_dir, rand_file)
        return np.load(f)

    def _reset(self):
        # self.map_details needs to be set before calling this
        # Either this is already set during construction, or subclass will call self._get_random_map and save
        assert hasattr(self, "map_details") and self.map_details is not None
        self._reset_internal_metrics()

        # Disable the internal logger
        logging.getLogger().setLevel(logging.WARNING)

        # Convert stored map array into input string representation
        map_str = hiddencell_to_mapstr(self.map_details["map_id"], self._max_steps)
        env_configs = {"grid": map_str, "obs_show_ids": False, "reward_structure": self._env_mode}
        self._env = rl_environment.Environment("stones_and_gems", **env_configs)

        # Return logging to previous state
        logging.getLogger().setLevel(logging.INFO)
        # Return current time step observation
        return self._timestep_to_state(self._env.reset())

    def _get_reward(self):
        # Subclass environments can modifiy just this method which determines how the
        # reward should be calculated from the next timestep received from the env
        return self._next_timestep.rewards[0]

    def num_actions(self):
        return self._num_actions

    def obs_shape(self):
        return self._env.game.observation_tensor_shape()

    def reset(self):
        self._reset()

    def step(self, action):
        assert action >= 0 and action < self._num_actions
        self._step += 1
        a = action + 1 if self._num_actions == 4 else action
        self._next_timestep = self._env.step([a])
        self._done = self._next_timestep.last() or self._step >= self._max_steps
        next_state = self._timestep_to_state(self._next_timestep) if not self._done else None
        reward = self._get_reward()
        self._cumulative_reward += reward
        return next_state, reward, self._done

    def did_win(self):
        return self._done and self._step < self._max_steps

    def is_done(self):
        return self._done

    @staticmethod
    def state_to_tensor(state):
        return torch.from_numpy(state).float().unsqueeze(0)

    def get_current_state(self):
        return self._timestep_to_state(self._env.get_time_step())

    def state_to_image(self, state=None):
        # (h, w, c)
        state_img = self.render(state)
        # Process a (h, w, c) image into a (c, h, w) image (static shape for nets)
        state_img = cv2.resize(state_img, dsize=(self._tensor_width, self._tensor_height), interpolation=cv2.INTER_NEAREST)
        state_img = np.transpose(state_img, (2, 0, 1))
        return state_img

    def render(self, state=None):
        # (h, w, c)
        if state is None:
            state = self._timestep_to_state(self._env.get_time_step())
        state_img = rnd_state_to_img(state)
        # Resize if necessary
        if self._render_width != -1 and self._render_height != -1:
            state_img = cv2.resize(state_img, dsize=(self._tensor_width, self._tensor_height), interpolation=cv2.INTER_NEAREST)
        return state_img

    def get_agent_index(self):
        base_obs_shape = self._env.game.observation_tensor_shape()
        state = np.array(self._env.get_time_step().observations["info_state"][0]).reshape(base_obs_shape)
        state = state[tilestr_to_visiblecellid["agent"]]
        idx = np.array(np.where(state != 0)).flatten()
        return tuple(idx)

    def get_exit_index(self):
        base_obs_shape = self._env.game.observation_tensor_shape()
        state = np.array(self._env.get_time_step().observations["info_state"][0]).reshape(base_obs_shape)
        state = np.add(state[tilestr_to_visiblecellid["exit_closed"]], state[tilestr_to_visiblecellid["exit_open"]])
        idx = np.array(np.where(state != 0)).flatten()
        return tuple(idx)
