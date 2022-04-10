import sys
import os
import hashlib
from copy import deepcopy
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from util.rnd_definitions import *
from rnd_py.rnd_game_util import *


# Default game parameters
kDefaultGameParams = {
    "obs_show_ids": False,  # Flag to show object ids in observation instead of binary channels
    "magic_wall_steps": 140,  # Number of steps before magic wall expire
    "blob_chance": 20,  # Chance to spawn another blob (out of 256)
    "blob_max_percentage": 0.16,  # Max number of blobs before they collapse (percentage of map size)
    "rng_seed": 0,  # Seed for anything that uses the rng
}


class RNDGameState:
    def __init__(self, game_params: dict):
        if "grid" not in game_params:
            print("Error: constructor requires grid param.")
            raise ValueError
        # overwrite param defaults with user provided params
        params = deepcopy(kDefaultGameParams)
        for k, v in game_params.items():
            params[k] = v

        # self._params = params

        # Set members
        self.reset(params)

    def _parse_grid(self, params) -> None:
        splt_symbol = "," if params["grid"].count(",") > 0 else "|"
        lines = [list(map(int, s.split(splt_symbol))) for s in params["grid"].split("\n")]
        assert len(lines[0]) == 4
        self._cols, self._rows, self._gems_required = lines[0][0], lines[0][1], lines[0][3]
        self._max_steps = lines[0][2] if lines[0][2] > 0 else None
        self._blob_max_size = params["blob_max_percentage"] * self._cols * self._rows

        # Create grid
        lines = lines[1:]
        if self._obs_show_ids:
            self._grid = np.zeros((NUM_HIDDEN_CELL_TYPE, self._rows, self._cols), dtype=np.uint16)
        else:
            self._grid = np.zeros((NUM_HIDDEN_CELL_TYPE, self._rows, self._cols), dtype=bool)
        self._has_updated = np.zeros((self._rows, self._cols), dtype=bool)
        assert len(lines) == self._rows
        for r, line in enumerate(lines):
            for c in range(self._cols):
                assert len(line) == self._cols
                self._has_updated[r, c] = False
                if line[c] == HiddenCellType.kEmpty or line[c] == HiddenCellType.kDirt:
                    self._grid[line[c], r, c] = 1
                else:
                    self._increment_counter()
                    self._grid[line[c], r, c] = self._id_counter

        self._unpacked_size = self._grid.size
        self._unpacked_shape = self._grid.shape
        self._pack_grid()

    def _pack_grid(self):
        if not self._obs_show_ids:
            self._grid = np.packbits(self._grid, axis=None)

    def _unpack_grid(self):
        if not self._obs_show_ids:
            self._grid = np.unpackbits(self._grid, count=self._unpacked_size).reshape(self._unpacked_shape).view(bool)

    def _increment_counter(self):
        if self._obs_show_ids:
            self._id_counter += 1
        else:
            self._id_counter = 1

    def _check_channel(self, coord: Tuple[int, int]) -> bool:
        channel_index = np.where(self._grid[(slice(None), *coord)] > 0)[0]
        return channel_index.size == 1

    def _grid_to_channel(self, coord: Tuple[int, int]) -> int:
        channel_index = np.where(self._grid[(slice(None), *coord)] > 0)[0]
        assert channel_index.size == 1
        return channel_index.item()

    def _grid_to_element(self, coord: Tuple[int, int]) -> Element:
        return kHiddenCellTypeToElement[HiddenCellType(self._grid_to_channel(coord))]

    def _grid_to_id(self, coord: Tuple[int, int]) -> int:
        channel = self._grid_to_channel(coord)
        return self._grid[(channel, *coord)]

    def _in_bounds(self, coord: Tuple[int, int], action: Directions = Directions.kNone) -> bool:
        row, col = coord_from_action(coord, action)
        return col >= 0 and col < self._cols and row >= 0 and row < self._rows

    def _is_type(self, coord: Tuple[int, int], element: Element, action: Directions = Directions.kNone) -> bool:
        new_coord = coord_from_action(coord, action)
        return self._in_bounds(coord, action) and self._grid_to_element(new_coord) == element

    def _has_property(self, coord: Tuple[int, int], property: ElementProperties, action: Directions = Directions.kNone) -> bool:
        new_coord = coord_from_action(coord, action)
        return self._in_bounds(coord, action) and (self._grid_to_element(new_coord).properties & property) > 0

    def _move_item(self, coord: Tuple[int, int], action: Directions) -> None:
        new_coord = coord_from_action(coord, action)
        channel_old = self._grid_to_channel(coord)
        channel_new = self._grid_to_channel(new_coord)
        self._grid[(channel_old, *new_coord)] = self._grid[(channel_old, *coord)]  # Move item
        self._grid[(channel_old, *coord)] = 0  # Unset previous item in old coord
        self._grid[(channel_new, *new_coord)] = 0  # Unset previous item in new coord
        self._grid[(HiddenCellType.kEmpty, *coord)] = 1  # Set previous coord to empty
        self._has_updated[new_coord] = True
        assert self._check_channel(coord) == True
        assert self._check_channel(new_coord) == True

    def _set_item(self, coord: Tuple[int, int], element: Element, id: int, action: Directions = Directions.kNone) -> None:
        new_coord = coord_from_action(coord, action)
        old_channel = self._grid_to_channel(new_coord)
        self._grid[(old_channel, *new_coord)] = 0  # Need to ensure we remove item already existing here
        new_channel = int(element.cell_type)
        self._grid[(new_channel, *new_coord)] = id
        assert self._check_channel(new_coord) == True  # Ensure exactly 1 channel is set

    def _get_item(self, coord: Tuple[int, int], action: Directions = Directions.kNone) -> Element:
        new_coord = coord_from_action(coord, action)
        return self._grid_to_element(new_coord)

    def _get_id(self, coord: Tuple[int, int], action: Directions = Directions.kNone) -> int:
        new_coord = coord_from_action(coord, action)
        return self._grid_to_id(new_coord)

    def _is_type_adjacent(self, coord: Tuple[int, int], element: Element) -> bool:
        return (
            self._is_type(coord, element, Directions.kUp)
            or self._is_type(coord, element, Directions.kLeft)
            or self._is_type(coord, element, Directions.kDown)
            or self._is_type(coord, element, Directions.kRight)
        )

    def _can_roll_left(self, coord: Tuple[int, int]) -> bool:
        return (
            self._has_property(coord, ElementProperties.kRounded, Directions.kDown)
            and self._is_type(coord, kElEmpty, Directions.kLeft)
            and self._is_type(coord, kElEmpty, Directions.kDownLeft)
        )

    def _can_roll_right(self, coord: Tuple[int, int]) -> bool:
        return (
            self._has_property(coord, ElementProperties.kRounded, Directions.kDown)
            and self._is_type(coord, kElEmpty, Directions.kRight)
            and self._is_type(coord, kElEmpty, Directions.kDownRight)
        )

    def _roll_left(self, coord: Tuple[int, int], element: Element) -> None:
        self._set_item(coord, element, self._get_id(coord))
        self._move_item(coord, Directions.kLeft)

    def _roll_right(self, coord: Tuple[int, int], element: Element) -> None:
        self._set_item(coord, element, self._get_id(coord))
        self._move_item(coord, Directions.kRight)

    def _push(self, coord: Tuple[int, int], stationary: Element, falling: Element, action: Directions) -> None:
        new_coord = coord_from_action(coord, action)
        if self._is_type(new_coord, kElEmpty, action):
            # Check if the element will become stationary or falling
            next_coord = coord_from_action(new_coord, action)
            is_empty = self._is_type(next_coord, kElEmpty, Directions.kDown)
            self._set_item(new_coord, falling if is_empty else stationary, self._grid_to_id(new_coord), action)
            self._move_item(coord, action)  # Move agent

    def _move_through_magic(self, coord: Tuple[int, int], element: Element) -> None:
        # Check if magic wall is still active
        if self._magic_wall_steps <= 0:
            return

        self._magic_active = True
        coord_below = coord_from_action(coord, Directions.kDown)
        # Need to ensure cell below magic wall is empty (so item can pass through)
        if self._is_type(coord_below, kElEmpty, Directions.kDown):
            self._set_item(coord, kElEmpty, 1)  # Empty and dirt ids are 1
            self._increment_counter()
            self._set_item(coord_below, element, self._id_counter, Directions.kDown)  # Spawned element gets new id

    def _explode(self, coord: Tuple[int, int], element: Element, action: Directions = Directions.kNone) -> None:
        new_coord = coord_from_action(coord, action)
        old_element = self._get_item(new_coord)
        exploded_element = kElementToExplosion[old_element] if old_element in kElementToExplosion else kElExplosionEmpty
        self._increment_counter()
        self._set_item(new_coord, element, self._id_counter)

        # Recursively check all directions for chain explosions
        for direction in Directions:
            if direction == Directions.kNone or not self._in_bounds(new_coord, direction):
                continue
            if self._has_property(new_coord, ElementProperties.kCanExplode, direction):
                self._explode(new_coord, exploded_element, direction)
            elif self._has_property(new_coord, ElementProperties.kConsumable, direction):
                self._increment_counter()
                self._set_item(new_coord, exploded_element, self._id_counter, direction)

    def _open_gate(self, el_gate_closed: Element) -> None:
        el_gate_open = kGateOpenMap[el_gate_closed]
        closed_gate_indices = np.transpose((self._grid[int(el_gate_closed.cell_type), :, :]).nonzero())
        # Convert closed gates to open
        for idx in closed_gate_indices:
            coord = (idx[0].item(), idx[1].item())
            self._set_item(coord, el_gate_open, self._grid_to_id(coord))

    def _update_stone(self, coord: Tuple[int, int]) -> None:
        if self._is_type(coord, kElEmpty, Directions.kDown):  # Set to falling
            self._set_item(coord, kElStoneFalling, self._grid_to_id(coord))
            self._update_stone_falling(coord)
        elif self._can_roll_left(coord):  # Roll left
            self._roll_left(coord, kElStoneFalling)
        elif self._can_roll_right(coord):  # Roll right
            self._roll_right(coord, kElStoneFalling)

    def _update_stone_falling(self, coord: Tuple[int, int]) -> None:
        if self._is_type(coord, kElEmpty, Directions.kDown):  # Continue to fall
            self._move_item(coord, Directions.kDown)
        elif self._has_property(coord, ElementProperties.kCanExplode, Directions.kDown):  # Falling stones explode items
            old_element = self._get_item(coord, Directions.kDown)
            exploded_element = kElementToExplosion[old_element] if old_element in kElementToExplosion else kElExplosionEmpty
            self._explode(coord, exploded_element, Directions.kDown)
        elif self._is_type(coord, kElWallMagicOn, Directions.kDown) or self._is_type(
            coord, kElWallMagicDormant, Directions.kDown
        ):  # Convert item through magic wall
            self._move_through_magic(coord, kMagicWallConversion[self._get_item(coord)])
        elif self._is_type(coord, kElNut, Directions.kDown):  # Falling on nut -> diamond
            self._increment_counter()
            self._set_item(coord, kElDiamond, self._id_counter, Directions.kDown)
            self._reward_signal |= RewardCodes.kRewardNutToDiamond
        elif self._is_type(coord, kElBomb, Directions.kDown):  # Falling on bomb -> explode
            old_element = self._get_item(coord, Directions.kDown)
            exploded_element = kElementToExplosion[old_element] if old_element in kElementToExplosion else kElExplosionEmpty
            self._explode(coord, exploded_element, Directions.kDown)
        elif self._can_roll_left(coord):  # Roll left
            self._roll_left(coord, kElStoneFalling)
        elif self._can_roll_right(coord):  # Roll right
            self._roll_right(coord, kElStoneFalling)
        else:  # Default option is for falling stone to become stationary
            self._set_item(coord, kElStone, self._grid_to_id(coord))

    def _update_diamond(self, coord: Tuple[int, int]) -> None:
        if self._is_type(coord, kElEmpty, Directions.kDown):  # Set to falling
            self._set_item(coord, kElDiamondFalling, self._grid_to_id(coord))
            self._update_diamond_falling(coord)
        elif self._can_roll_left(coord):  # Roll left
            self._roll_left(coord, kElDiamondFalling)
        elif self._can_roll_right(coord):  # Roll right
            self._roll_right(coord, kElDiamondFalling)

    def _update_diamond_falling(self, coord: Tuple[int, int]) -> None:
        if self._is_type(coord, kElEmpty, Directions.kDown):  # Continue to fall
            self._move_item(coord, Directions.kDown)
        elif (
            self._has_property(coord, ElementProperties.kCanExplode, Directions.kDown)
            and not self._is_type(coord, kElBomb, Directions.kDown)
            and not self._is_type(coord, kElBombFalling, Directions.kDown)
        ):  # Falling diamond explode items (but not bombs)
            old_element = self._get_item(coord, Directions.kDown)
            exploded_element = kElementToExplosion[old_element] if old_element in kElementToExplosion else kElExplosionEmpty
            self._explode(coord, exploded_element, Directions.kDown)
        elif self._is_type(coord, kElWallMagicOn, Directions.kDown) or self._is_type(
            coord, kElWallMagicDormant, Directions.kDown
        ):  # Convert item through magic wall
            self._move_through_magic(coord, kMagicWallConversion[self._get_item(coord)])
        elif self._can_roll_left(coord):  # Roll left
            self._roll_left(coord, kElDiamondFalling)
        elif self._can_roll_right(coord):  # Roll right
            self._roll_right(coord, kElDiamondFalling)
        else:  # Default option is for falling diamond to become stationary
            self._set_item(coord, kElDiamond, self._grid_to_id(coord))

    def _update_nut(self, coord: Tuple[int, int]) -> None:
        if self._is_type(coord, kElEmpty, Directions.kDown):  # Set to falling
            self._set_item(coord, kElNutFalling, self._grid_to_id(coord))
            self._update_nut_falling(coord)
        elif self._can_roll_left(coord):  # Roll left
            self._roll_left(coord, kElNutFalling)
        elif self._can_roll_right(coord):  # Roll right
            self._roll_right(coord, kElNutFalling)

    def _update_nut_falling(self, coord: Tuple[int, int]) -> None:
        if self._is_type(coord, kElEmpty, Directions.kDown):  # Continue to fall
            self._move_item(coord, Directions.kDown)
        elif self._can_roll_left(coord):  # Roll left
            self._roll_left(coord, kElNutFalling)
        elif self._can_roll_right(coord):  # Roll right
            self._roll_right(coord, kElNutFalling)
        else:  # Default option is for falling nut to become stationary
            self._set_item(coord, kElNut, self._grid_to_id(coord))

    def _update_bomb(self, coord: Tuple[int, int]) -> None:
        if self._is_type(coord, kElEmpty, Directions.kDown):  # Set to falling
            self._set_item(coord, kElBombFalling, self._grid_to_id(coord))
            self._update_bomb_falling(coord)
        elif self._can_roll_left(coord):  # Roll left
            self._roll_left(coord, kElBombFalling)
        elif self._can_roll_right(coord):  # Roll right
            self._roll_right(coord, kElBombFalling)

    def _update_bomb_falling(self, coord: Tuple[int, int]) -> None:
        if self._is_type(coord, kElEmpty, Directions.kDown):  # Continue to fall
            self._move_item(coord, Directions.kDown)
        elif self._can_roll_left(coord):  # Roll left
            self._roll_left(coord, kElBombFalling)
        elif self._can_roll_right(coord):  # Roll right
            self._roll_right(coord, kElBombFalling)
        else:  # Default option is for falling bomb is to explode
            old_element = self._get_item(coord)
            exploded_element = kElementToExplosion[old_element] if old_element in kElementToExplosion else kElExplosionEmpty
            self._explode(coord, exploded_element)

    def _update_exit(self, coord: Tuple[int, int]) -> None:
        # Open exit if enough gems collected
        if self._gems_collected >= self._gems_required:
            self._set_item(coord, kElExitOpen, self._grid_to_id(coord))

    def _update_agent(self, coord: Tuple[int, int], action: Directions) -> None:
        if self._is_type(coord, kElEmpty, action) or self._is_type(coord, kElDirt, action):  # Move if empty/dirt
            self._move_item(coord, action)
        elif self._is_type(coord, kElDiamond, action) or self._is_type(coord, kElDiamondFalling, action):  # Collect gems
            self._gems_collected += 1
            self._current_reward += kGemPoints[self._get_item(coord, action)]
            self._reward_signal |= RewardCodes.kRewardCollectDiamond
            self._move_item(coord, action)
        elif IsActionHorz(action) and (
            self._is_type(coord, kElStone, action)
            or self._is_type(coord, kElNut, action)
            or self._is_type(coord, kElBomb, action)
        ):  # Push stone, nut, or bomb horizontal
            self._push(coord, self._get_item(coord, action), kElToFalling[self._get_item(coord, action)], action)
        elif IsKey(self._get_item(coord, action)):  # Collecting key, set gate open
            self._open_gate(kKeyToGate[self._get_item(coord, action)])
            self._move_item(coord, action)
            self._reward_signal |= RewardCodes.kRewardCollectKey
        elif IsOpenGate(self._get_item(coord, action)):  # Walking through open gate
            coord_gate = coord_from_action(coord, action)
            if self._has_property(coord_gate, ElementProperties.kTraversable, action):
                if self._is_type(coord_gate, kElDiamond, action):  # Could pass through onto diamond
                    self._gems_collected += 1
                    self._current_reward += kGemPoints[kElDiamond]
                    self._reward_signal |= RewardCodes.kRewardCollectDiamond
                elif IsKey(self._get_item(coord_gate, action)):  # Could pass through onto key
                    self._open_gate(kKeyToGate[self._get_item(coord_gate, action)])
                    self._reward_signal |= RewardCodes.kRewardCollectKey
                self._set_item(coord_gate, kElAgent, self._grid_to_id(coord), action)
                self._set_item(coord, kElEmpty, 1)
                self._reward_signal |= RewardCodes.kRewardWalkThroughGate
        elif self._is_type(coord, kElExitOpen, action):  # Walking though exit
            self._move_item(coord, action)
            self._set_item(coord, kElAgentInExit, self._grid_to_id(coord), action)  # Different from open_spiel
            self._current_reward += self._steps_remaining if self._steps_remaining is not None else kGemPoints[kElAgentInExit]
            self._reward_signal |= RewardCodes.kRewardWalkThroughExit

    def _update_firefly(self, coord: Tuple[int, int], action: Directions) -> None:
        new_direction = kRotateLeft[action]
        if self._is_type_adjacent(coord, kElAgent) or self._is_type_adjacent(coord, kElBlob):  # Exploide if touching agent/blob
            old_element = self._get_item(coord)
            exploded_element = kElementToExplosion[old_element] if old_element in kElementToExplosion else kElExplosionEmpty
            self._explode(coord, exploded_element)
        elif self._is_type(coord, kElEmpty, new_direction):  # First try to rotate left
            self._set_item(coord, kDirectionToFirefly[new_direction], self._grid_to_id(coord))
            self._move_item(coord, new_direction)
        elif self._is_type(coord, kElEmpty, action):  # Then try to move forward
            self._set_item(coord, kDirectionToFirefly[action], self._grid_to_id(coord))
            self._move_item(coord, action)
        else:  # No other options, rotate right
            self._set_item(coord, kDirectionToFirefly[kRotateRight[action]], self._grid_to_id(coord))

    def _update_butterfly(self, coord: Tuple[int, int], action: Directions) -> None:
        new_direction = kRotateRight[action]
        if self._is_type_adjacent(coord, kElAgent) or self._is_type_adjacent(coord, kElBlob):  # Exploide if touching agent/blob
            old_element = self._get_item(coord)
            exploded_element = kElementToExplosion[old_element] if old_element in kElementToExplosion else kElExplosionEmpty
            self._explode(coord, exploded_element)
        elif self._is_type(coord, kElEmpty, new_direction):  # First try to rotate right
            self._set_item(coord, kDirectionToButterfly[new_direction], self._grid_to_id(coord))
            self._move_item(coord, new_direction)
        elif self._is_type(coord, kElEmpty, action):  # Then try to move forward
            self._set_item(coord, kDirectionToButterfly[action], self._grid_to_id(coord))
            self._move_item(coord, action)
        else:  # No other options, rotate left
            self._set_item(coord, kDirectionToButterfly[kRotateLeft[action]], self._grid_to_id(coord))

    def _update_orange(self, coord: Tuple[int, int], action: Directions) -> None:
        if self._is_type(coord, kElEmpty, action):  # Continue moving in direction
            self._move_item(coord, action)
        elif self._is_type_adjacent(coord, kElAgent):  # Run into agent -> explode
            old_element = self._get_item(coord)
            exploded_element = kElementToExplosion[old_element] if old_element in kElementToExplosion else kElExplosionEmpty
            self._explode(coord, exploded_element)
        else:  # Blocked, roll for new direction
            open_directions = [
                direction
                for direction in Directions
                if (
                    direction != Directions.kNone
                    and self._in_bounds(coord, direction)
                    and self._is_type(coord, kElEmpty, direction)
                )
            ]
            # Roll for new direction
            if len(open_directions) > 0:
                new_direction = open_directions[self._rng.choice(len(open_directions))]
                self._set_item(coord, kDirectionToOrange[new_direction], self._grid_to_id(coord))

    def _update_magic_wall(self, coord: Tuple[int, int]) -> None:
        if self._magic_active:  # Dormant
            self._set_item(coord, kElWallMagicOn, self._grid_to_id(coord))
        elif self._magic_wall_steps > 0:  # Active
            self._set_item(coord, kElWallMagicDormant, self._grid_to_id(coord))
        else:  # Expired
            self._set_item(coord, kElWallMagicExpired, self._grid_to_id(coord))

    def _update_blob(self, coord: Tuple[int, int]) -> None:
        if self._blob_swap != kNullElement:  # Replace blob if swap element set
            self._increment_counter()
            self._set_item(coord, self._blob_swap, self._id_counter)
        else:
            self._blob_size += 1
            if self._is_type_adjacent(coord, kElEmpty) or self._is_type_adjacent(
                coord, kElDirt
            ):  # Check if at least one tile blob can grow
                self._blob_enclosed = False
            # Roll if to grow and direction
            will_grow = self._rng.integers(0, 255) < self._blob_chance
            possible_directions = [Directions.kUp, Directions.kLeft, Directions.kDown, Directions.kRight]
            direction_grow = possible_directions[self._rng.choice(len(possible_directions))]
            if will_grow and (self._is_type(coord, kElEmpty, direction_grow) or self._is_type(coord, kElDirt, direction_grow)):
                self._increment_counter()
                self._set_item(coord, kElBlob, self._id_counter, direction_grow)

    def _update_explosions(self, coord: Tuple[int, int]) -> None:
        self._increment_counter()
        if kExplosionToElement[self._get_item(coord)] == kElDiamond:
            self._reward_signal |= RewardCodes.kRewardButterflyToDiamond
        self._set_item(coord, kExplosionToElement[self._get_item(coord)], self._id_counter)

    def _start_scan(self) -> None:
        # Update global flags
        if self._steps_remaining is not None:
            self._steps_remaining += -1
        self._current_reward = 0.0
        self._blob_size = 0
        self._blob_enclosed = True
        self._reward_signal = 0
        # Reset elements
        self._has_updated[:] = False
        self._unpack_grid()

    def _end_scan(self) -> None:
        if self._blob_swap == kNullElement:  # Check if blob status
            if self._blob_enclosed:  # If enclosed, it becomes diamonds
                self._blob_swap = kElDiamond
            elif self._blob_size > self._blob_max_size:  # If blob too large, it becomes stones
                self._blob_swap = kElStone
        if self._magic_active:  # Reduce magic wall steps if active
            self._magic_wall_steps = max(self._magic_wall_steps - 1, 0)
        # Check if still active
        self._magic_active = self._magic_active and self._magic_wall_steps > 0
        self._pack_grid()

    def reset(self, params) -> None:
        """Reset the state to the beginning"""
        self._magic_wall_steps = params["magic_wall_steps"]
        self._magic_active = False
        self._blob_size = 0
        self._blob_chance = params["blob_chance"]
        self._blob_enclosed = False
        self._blob_swap = kNullElement
        self._gems_collected = 0
        self._current_reward = 0
        self._obs_show_ids = params["obs_show_ids"]
        self._id_counter = 1
        self._seed = params["rng_seed"]
        self._rng = np.random.default_rng(self._seed)
        self._parse_grid(params)
        self._steps_remaining = self._max_steps
        self._reward_signal = 0

    def apply_action(self, action: int) -> None:
        """Perform the action and step the state forward one step

        Args:
            actions: Integer action code to apply
        """
        assert action >= 0 and action < NUM_ACTIONS
        self._start_scan()

        # Find where agent is and update its position
        agent_idx = np.where(self._grid[int(HiddenCellType.kAgent), :, :])
        coord = (agent_idx[0].item(), agent_idx[1].item())
        self._update_agent(coord, Directions(action))

        # Check each cell and apply respective dynamics function
        for r in range(self._rows):
            for c in range(self._cols):
                element = self._get_item((r, c))
                if self._has_updated[r, c]:
                    continue
                elif element == kElStone:
                    self._update_stone((r, c))
                elif element == kElStoneFalling:
                    self._update_stone_falling((r, c))
                elif element == kElDiamond:
                    self._update_diamond((r, c))
                elif element == kElDiamondFalling:
                    self._update_diamond_falling((r, c))
                elif element == kElNut:
                    self._update_nut((r, c))
                elif element == kElNutFalling:
                    self._update_nut_falling((r, c))
                elif element == kElBomb:
                    self._update_bomb((r, c))
                elif element == kElBombFalling:
                    self._update_bomb_falling((r, c))
                elif element == kElExitClosed:
                    self._update_exit((r, c))
                elif IsButterfly(element):
                    self._update_butterfly((r, c), kButterflyToDirection[element])
                elif IsFirefly(element):
                    self._update_firefly((r, c), kFireflyToDirection[element])
                elif IsOrange(element):
                    self._update_orange((r, c), kOrangeToDirection[element])
                elif IsMagicWall(element):
                    self._update_magic_wall((r, c))
                elif element == kElBlob:
                    self._update_blob((r, c))
                elif IsExplosion(element):
                    self._update_explosions((r, c))

        self._end_scan()

    def is_terminal(self) -> bool:
        """Return True if the game is over, false otherwise."""
        self._unpack_grid()
        out_of_time = self._steps_remaining is not None and self._steps_remaining <= 0
        result = out_of_time or np.where(self._grid[int(HiddenCellType.kAgent), :, :])[0].size == 0
        self._pack_grid()
        return result

    def is_solution(self) -> bool:
        """Return True if the game is solved, false otherwise."""
        self._unpack_grid()
        out_of_time = self._steps_remaining is not None and self._steps_remaining <= 0
        result = not out_of_time and np.where(self._grid[int(HiddenCellType.kAgentInExit), :, :])[0].size == 1
        self._pack_grid()
        return result

    def get_observation(self) -> np.ndarray:
        """Get the current observation as an numpy array"""
        self._unpack_grid()
        obs = np.zeros((NUM_VISIBLE_CELL_TYPE, self._rows, self._cols), dtype=np.float32)
        for r in range(self._rows):
            for c in range(self._cols):
                grid_channel = self._grid_to_channel((r, c))
                obs_channel = HiddenToVisibleMapping[grid_channel]
                obs[obs_channel, r, c] = self._grid[grid_channel, r, c] if self._obs_show_ids else 1
        self._pack_grid()
        return obs

    def legal_actions(self) -> Tuple[int]:
        """Gets the current legal actions set"""
        return [] if self.is_terminal() else [int(a) for a in Actions]

    def observation_shape(self) -> Tuple[int]:
        """Get the observation shape"""
        return (NUM_VISIBLE_CELL_TYPE, self._rows, self._cols)

    def get_reward_signal(self) -> int:
        """Get the current reward signal"""
        return self._reward_signal

    def __hash__(self) -> int:
        v = self.get_observation().view(np.uint8)
        # return hash(hashlib.sha1(v).hexdigest())
        return int("0x{}".format(hashlib.sha1(v).hexdigest()), 0)


def main():
    print("Paste map string: ")
    sentinel = ""
    map_str = "\n".join(iter(input, sentinel))
    env = RNDGameState({"grid": map_str})
    NUM_STEPS = 1000

    print(hash(env))

    # start = time.time()
    # for _ in range(NUM_STEPS):
    #     env.apply_action(0)
    #     obs = env.get_observation()
    # end = time.time()
    # duration = end - start
    # print("Total time for {} steps: {:.4f}s".format(NUM_STEPS, duration))
    # print("Time per step : {:.4f}s".format(duration / NUM_STEPS))


if __name__ == "__main__":
    main()
