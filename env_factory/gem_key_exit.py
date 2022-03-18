import os
import sys
import itertools
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from util.rnd_util import *
from util.rnd_util import _random_choice


# Always have 4 rooms
# But set number of keys
# All rooms have either open or closed door
# Keys are set linearly, or randomly if door already open?

MAX_ROOMS = 4

def create_gem_key_exit(
    size: int = 10,
    num_gems: int = 0,
    seed: int = 0,
    num_rooms: int = 0,
    room_size: int = 6,
    num_locked_doors: int = 0,
    num_keys_in_main: int = 0,
    ratio_gems_in_room: float = 0,
    keys_in_order: bool = True,
    max_steps: int = 9999
):
    # Empty map
    rng = np.random.default_rng(seed)
    m = create_empty_map(size, gen=rng)

    # Create any rooms
    assert num_rooms >= 0 and num_rooms <= 4
    assert room_size > 0 and room_size < (size - 1) / 2
    rooms = [create_empty_room(room_size, room_size, gen=rng) for _ in range(MAX_ROOMS)]
    room_positions = get_room_positions_corner(MAX_ROOMS, gen=rng)
    room_offsets = [get_room_offset_corner(m, r, p) for r, p in zip(rooms, room_positions)]
    blocked_idxs = list(itertools.chain(*[get_blocked_idx_corner(m, r, p) for r, p in zip(rooms, room_positions)]))

    # Keys and doors ordering
    assert num_locked_doors <= 4
    keys, doors_closed, doors_open = get_shuffled_keys_doors(gen=rng)
    exit_type = HiddenCellType.kExitClosed if num_gems > 0 else HiddenCellType.kExitOpen
    items_in_room = {
        0 : [exit_type] if keys_in_order else [exit_type, keys[2]],
        1 : [keys[0]],
        2 : [keys[1]],
        3 : [keys[2]] if keys_in_order else [],
    }

    # Add keys/doors in items
    # Keys in order                 Keys not in order
    # r1 (exit, d1)                 r1 (exit & k3, d1) 
    # r2 (k1, d2)                   r2 (k1, d2)
    # r3 (k2, d3)                   r3 (k2, d3)
    # r4 (k3, d4)                   r4 (, d4)
    # for i in range(num_rooms):
    for i in range(MAX_ROOMS):
        door = doors_closed[i] if i < num_locked_doors else doors_open[i]
        add_item_border_corner(rooms[i], door, room_positions[i], gen=rng)
        for item in items_in_room[i]:
            add_item_inside_room(rooms[i], item, gen=rng)

    # Places gems
    assert ratio_gems_in_room >= 0 and ratio_gems_in_room <= 1
    num_gems_in_rooms = int(num_gems * ratio_gems_in_room) if num_rooms > 0 else 0
    for _ in range(num_gems_in_rooms):
        while True:
            try:
                room_idx = _random_choice([i for i in range(num_rooms)], gen=rng)
                add_item_inside_room(rooms[room_idx], HiddenCellType.kDiamond, gen=rng)
                break
            except:
                pass

    # Put rooms in map
    for r, o in zip(rooms, room_offsets):
        add_room_to_map(m, r, o)

    # Place random keys in main (tests if we can learn to ignore)
    for i in range(num_keys_in_main):
        key =  _random_choice(keys, gen=rng)
        add_item_inside_room(m, key, blocked_tiles=blocked_idxs, gen=rng)

    # Place remaining gems in main portion of map
    for _ in range(num_gems - num_gems_in_rooms):
        add_item_inside_room(m, HiddenCellType.kDiamond, blocked_tiles=blocked_idxs, gen=rng)

    # Place agent inside main
    add_item_inside_room(m, HiddenCellType.kAgent, blocked_tiles=blocked_idxs, gen=rng)

    return map_to_str(m, max_steps=max_steps, num_gems=num_gems)


if __name__ == "__main__":
    mapstr = create_gem_key_exit()


