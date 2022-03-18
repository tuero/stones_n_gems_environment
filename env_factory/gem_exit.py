import os
import sys
import itertools
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from util.rnd_util import *
from util.rnd_util import _random_choice


def create_env_gem_exit(
    size: int = 10,
    num_gems: int = 0,
    seed: int = 0,
    num_rooms: int = 0,
    room_size: int = 6,
    ratio_gems_in_room: float = 0,
    exit_in_room: bool = False,
    max_steps: int = 9999
):
    # Empty map
    rng = np.random.default_rng(seed)
    m = create_empty_map(size, gen=rng)

    # Create any rooms
    assert num_rooms <= 4
    assert num_rooms == 0 or room_size > 0 and room_size < (size - 1) / 2
    rooms = [create_empty_room(room_size, room_size, gen=rng) for _ in range(num_rooms)]
    room_positions = get_room_positions_corner(num_rooms, gen=rng)
    room_offsets = [get_room_offset_corner(m, r, p) for r, p in zip(rooms, room_positions)]
    blocked_idxs = list(itertools.chain(*[get_blocked_idx_corner(m, r, p) for r, p in zip(rooms, room_positions)]))

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

    # Add exit in room
    if exit_in_room and num_rooms > 0:
        while True:
            try:
                room_idx = _random_choice([i for i in range(num_rooms)], gen=rng)
                add_item_inside_room(rooms[room_idx], HiddenCellType.kExitClosed if num_gems > 0 else HiddenCellType.kExitOpen, gen=rng)
                break
            except:
                pass

    # Add empty space on room border so we can access
    for r, p in zip(rooms, room_positions):
        add_item_border_corner(r, HiddenCellType.kEmpty, p, gen=rng)

    # Put rooms in map
    for r, o in zip(rooms, room_offsets):
        add_room_to_map(m, r, o)

    # Place remaining gems in main portion of map
    for _ in range(num_gems - num_gems_in_rooms):
        add_item_inside_room(m, HiddenCellType.kDiamond, blocked_tiles=blocked_idxs, gen=rng)

    # Place agent and exit if we haven't already
    if not exit_in_room or num_rooms == 0:
        add_item_inside_room(m, HiddenCellType.kExitClosed if num_gems > 0 else HiddenCellType.kExitOpen, blocked_tiles=blocked_idxs, gen=rng)
    add_item_inside_room(m, HiddenCellType.kAgent, blocked_tiles=blocked_idxs, gen=rng)

    return map_to_str(m, max_steps=max_steps, num_gems=num_gems)


if __name__ == "__main__":
    mapstr = create_env_gem_exit(size=16, num_gems=5, num_rooms=2, room_size=6, ratio_gems_in_room=0.5, exit_in_room=True)
    print(mapstr)

