import os
import sys
import random
import copy
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from util.rnd_definitions import tilestr_to_hiddencellid, HiddenCellType
from util.rnd_definitions import rnd_background_tiles_hidden
from util.rnd_definitions import rnd_keys_hidden, rnd_doors_hidden, rnd_doorsopen_hidden

from typing import Tuple


def _random_choice(items: Tuple, gen: np.random.RandomState = None):
    return random.choice(items) if gen is None else items[gen.choice(len(items))]


def _random_shuffle(items: Tuple, gen: np.random.RandomState = None):
    random.shuffle(items) if gen is None else gen.shuffle(items)


def fill_room(room: np.ndarray, fill_tiles: Tuple[HiddenCellType], gen: np.random.RandomState = None):
    """Fill a room with a given tile.
    Args:
        room: The room numpy array
        gen: Generator for RNG
    """
    rows, cols = room.shape[0], room.shape[1]
    for r in range(1, rows - 1):
        for c in range(1, cols - 1):
            room[r, c] = _random_choice(fill_tiles, gen)


def add_item_inside_room(
    room: np.ndarray,
    tile_id: HiddenCellType,
    blocked_tiles: Tuple[Tuple[int, int]] = [],
    background_tiles: Tuple[HiddenCellType] = rnd_background_tiles_hidden,
    gen: np.random.RandomState = None,
):
    """Add an item in the interior of a room.
    Args:
        room: The room to insert into
        tile_id: HiddenCellType of the tile to add
        blocked_tiles: Indices to not add to
        background_tiles: Tiles which are considered background, and can be written over
        gen: Generator for RNG
    """
    rows, cols = room.shape[0], room.shape[1]
    room_indices = [
        (r, c)
        for r in range(2, rows - 2)
        for c in range(2, cols - 2)
        if room[r, c] in background_tiles and (r, c) not in blocked_tiles
    ]
    assert len(room_indices) > 0
    room_idx = _random_choice(room_indices, gen)
    room[room_idx[0], room_idx[1]] = tile_id


def add_item_border_room(room: np.ndarray, tile_id: HiddenCellType, room_position, gen: np.random.RandomState = None):
    rows, cols = room.shape[0], room.shape[1]
    room_indices = []
    room_indices += [(r, 0) for r in range(1, rows - 1)]
    room_indices += [(r, cols - 1) for r in range(1, rows - 1)]
    room_indices += [(0, c) for c in range(1, cols - 1)]
    room_indices += [(rows - 1, c) for c in range(1, cols - 1)]
    assert len(room_indices) > 0
    room_idx = _random_choice(room_indices, gen)
    room[room_idx[0], room_idx[1]] = tile_id


def add_item_border_corner(room: np.ndarray, tile_id: HiddenCellType, room_position: str, gen: np.random.RandomState = None):
    """Add an item along the border of a room that's placed in the corner of the map.
    Args:
        room: The room to insert into
        tile_id: HiddenCellType of the tile to add
        room_position: Position of the map the room is placed in
        gen: Generator for RNG
    """
    rows, cols = room.shape[0], room.shape[1]
    if room_position == "TOP_LEFT":
        idxs = [(rows - 1, c) for c in range(1, cols - 1)] + [(r, cols - 1) for r in range(1, rows - 1)]
    elif room_position == "TOP_RIGHT":
        idxs = [(rows - 1, c) for c in range(1, cols - 1)] + [(r, 0) for r in range(1, rows - 1)]
    elif room_position == "BOTTOM_LEFT":
        idxs = [(0, c) for c in range(1, cols - 1)] + [(r, cols - 1) for r in range(1, rows - 1)]
    elif room_position == "BOTTOM_RIGHT":
        idxs = [(0, c) for c in range(1, cols - 1)] + [(r, 0) for r in range(1, rows - 1)]

    idx = _random_choice(idxs, gen)
    room[idx[0], idx[1]] = tile_id


def add_room_to_map(m: np.ndarray, room: np.ndarray, room_offset: Tuple[int, int]):
    """Add the room to the map
    Args:
        m: The underlying map
        room: The room to insert
        room_offset: Starting index of the room (top left index)
        gen: Generator for RNG
    """
    start_r, start_c = room_offset
    rows, cols = room.shape[0], room.shape[1]
    for r in range(rows):
        for c in range(cols):
            m[r + start_r, c + start_c] = room[r, c]


def get_room_positions_corner(num_rooms: int, gen: np.random.RandomState = None) -> Tuple[str]:
    """Get room offsets for the rooms
    Args:
        num_rooms: Number of rooms
        gen: Generator for RNG

    Returns:
        Corners which rooms are placed in the map
    """
    corners = ["TOP_LEFT", "TOP_RIGHT", "BOTTOM_LEFT", "BOTTOM_RIGHT"]
    assert num_rooms <= len(corners)
    _random_shuffle(corners, gen)
    return [corners[i] for i in range(num_rooms)]


def get_room_offset_corner(m: np.ndarray, room: np.ndarray, room_position: str) -> Tuple[int, int]:
    """Get the room offset for a corner placed room
    Args:
        m: The underlying map
        room: The room to insert
        room_position: Position of the map the room is placed in
        gen: Generator for RNG

    Returns:
        Offset index for the room (top left index)
    """
    rows_room, cols_room = room.shape[0], room.shape[1]
    rows_m, cols_m = m.shape[0], m.shape[1]
    if room_position == "TOP_LEFT":
        start_r, start_c = 0, 0
    elif room_position == "TOP_RIGHT":
        start_r, start_c = 0, cols_m - cols_room
    elif room_position == "BOTTOM_LEFT":
        start_r, start_c = rows_m - rows_room, 0
    elif room_position == "BOTTOM_RIGHT":
        start_r, start_c = rows_m - rows_room, cols_m - cols_room
    else:
        raise ValueError
    return (start_r, start_c)


def get_blocked_idx_corner(m: np.ndarray, room: np.ndarray, room_position: str) -> Tuple[Tuple[int, int]]:
    """Get the list of blocked indices given a corner room position
    Args:
        m: The underlying map
        room: The room to insert
        room_position: Position of the map the room is placed in

    Returns:
        List of blocked tiles for the underlying map
    """
    start_r, start_c = get_room_offset_corner(m, room, room_position)
    rows_room, cols_room = room.shape[0], room.shape[1]
    return [(r + start_r, c + start_c) for r in range(rows_room) for c in range(cols_room)]


def _create_base_room(
    width: int, 
    height: int, 
    fill_tiles: Tuple[HiddenCellType] = rnd_background_tiles_hidden, 
    gen: np.random.RandomState = None
) -> np.ndarray:
    m = np.zeros((height, width), dtype=np.uint8)
    for r in range(height):
        m[r, 0] = tilestr_to_hiddencellid["wall_brick"]  # left column
        m[r, width - 1] = tilestr_to_hiddencellid["wall_brick"]  # right column
    for c in range(width):
        m[0, c] = tilestr_to_hiddencellid["wall_brick"]  # top row
        m[height - 1, c] = tilestr_to_hiddencellid["wall_brick"]  # bottom row
    fill_room(m, fill_tiles, gen)
    return m


def create_empty_room(
    room_width: int = 8,
    room_height: int = 8,
    fill_tiles: Tuple[HiddenCellType] = rnd_background_tiles_hidden,
    gen: np.random.RandomState = None,
) -> np.ndarray:
    """Create an empty square map.
    Args:
        room_width: The width of the room
        room_height: The height of the room
        fill_tiles: Background tiles to fill room
        gen: Generator for RNG

    Returns:
        An empty room
    """
    m = _create_base_room(room_width, room_height, fill_tiles=fill_tiles, gen=gen)
    return m


def create_empty_map(
    map_size: int, 
    fill_tiles: Tuple[HiddenCellType] = rnd_background_tiles_hidden, 
    gen: np.random.RandomState = None
) -> np.ndarray:
    """Create an empty square map.
    Args:
        map_size: The width/height of the room
        fill_tiles: Background tiles to fill room
        gen: Generator for RNG

    Returns:
        An empty map
    """
    m = _create_base_room(map_size, map_size, fill_tiles=fill_tiles, gen=gen)
    return m


def map_to_str(m, max_steps: int, num_gems: int) -> str:
    rows, cols = m.shape[0], m.shape[1]
    output_str = "{},{},{},{}\n".format(rows, cols, max_steps, num_gems)
    for r in range(rows):
        for c in range(cols):
            output_str += "{:02d},".format(m[r, c])
        output_str = output_str[:-1] + "\n"
    return output_str[:-1]


def flatten_map_str(map_str: str) -> str:
    return map_str.replace("\n", ",")

def pack_flat_map_str(flat_map_str: str) -> str:
    map_str_list = flat_map_str.split(",")
    output_str = ",".join(map_str_list[:4]) + "\n"
    rows, cols = int(map_str_list[0]), int(map_str_list[1])
    for r in range(rows):
        for c in range(cols):
            output_str += "{},".format(map_str_list[r*cols + c + 4])
        output_str = output_str[:-1] + "\n"
    return output_str[:-1]

def get_shuffled_keys_doors(gen: np.random.RandomState = None):
    keys = copy.deepcopy(rnd_keys_hidden)
    doors_closed = copy.deepcopy(rnd_doors_hidden)
    doors_open = copy.deepcopy(rnd_doorsopen_hidden)
    idx_shuffle = [i for i in range(len(keys))]
    _random_shuffle(idx_shuffle, gen=gen)
    keys = [keys[i] for i in idx_shuffle]
    doors_closed = [doors_closed[i] for i in idx_shuffle]
    doors_open = [doors_open[i] for i in idx_shuffle]
    return keys, doors_closed, doors_open
