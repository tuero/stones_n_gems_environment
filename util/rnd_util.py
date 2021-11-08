import os
import sys
import random
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from util.rnd_definitions import tilestr_to_hiddencellid, HiddenCellType
from util.rnd_definitions import rnd_background_tiles_hidden

from typing import Tuple


def fill_room(room: np.ndarray, fill_tiles: Tuple[HiddenCellType]):
    """Fill a room with a given tile.
    Args:
        room: The room numpy array
    """
    rows, cols = room.shape[0], room.shape[1]
    for r in range(1, rows - 1):
        for c in range(1, cols - 1):
            room[r, c] = random.choice(fill_tiles)


def add_item_inside_room(
    room: np.ndarray,
    tile_id: HiddenCellType,
    blocked_tiles: Tuple[Tuple[int, int]] = [],
    background_tiles: Tuple[HiddenCellType] = rnd_background_tiles_hidden,
):
    """Add an item in the interior of a room.
    Args:
        room: The room to insert into
        tile_id: HiddenCellType of the tile to add
        blocked_tiles: Indices to not add to
        background_tiles: Tiles which are considered background, and can be written over
    """
    rows, cols = room.shape[0], room.shape[1]
    room_indices = [
        (r, c)
        for r in range(2, rows - 2)
        for c in range(2, cols - 2)
        if room[r, c] in background_tiles and (r, c) not in blocked_tiles
    ]
    assert len(room_indices) > 0
    room_idx = random.choice(room_indices)
    room[room_idx[0], room_idx[1]] = tile_id


def add_item_border_room(room: np.ndarray, tile_id: HiddenCellType, room_position):
    rows, cols = room.shape[0], room.shape[1]
    room_indices = []
    room_indices += [(r, 0) for r in range(1, rows - 1)]
    room_indices += [(r, cols - 1) for r in range(1, rows - 1)]
    room_indices += [(0, c) for c in range(1, cols - 1)]
    room_indices += [(rows - 1, c) for c in range(1, cols - 1)]
    assert len(room_indices) > 0
    room_idx = random.choice(room_indices)
    room[room_idx[0], room_idx[1]] = tile_id


def add_item_border_corner(room: np.ndarray, tile_id: HiddenCellType, room_position: str):
    """Add an item along the border of a room that's placed in the corner of the map.
    Args:
        room: The room to insert into
        tile_id: HiddenCellType of the tile to add
        room_position: Position of the map the room is placed in
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

    idx = random.choice(idxs)
    room[idx[0], idx[1]] = tile_id


def add_room_to_map(m: np.ndarray, room: np.ndarray, room_offset: Tuple[int, int]):
    """Add the room to the map
    Args:
        m: The underlying map
        room: The room to insert 
        room_offset: Starting index of the room (top left index)
    """
    start_r, start_c = room_offset
    rows, cols = room.shape[0], room.shape[1]
    for r in range(rows):
        for c in range(cols):
            m[r + start_r, c + start_c] = room[r, c]


def get_room_positions_corner(num_rooms: int) -> Tuple[str]:
    """Get room offsets for the rooms
    Args:
        num_rooms: Number of rooms

    Returns:
        Corners which rooms are placed in the map
    """
    corners = ["TOP_LEFT", "TOP_RIGHT", "BOTTOM_LEFT", "BOTTOM_RIGHT"]
    assert num_rooms <= len(corners)
    random.shuffle(corners)
    return [corners[i] for i in range(num_rooms)]


def get_room_offset_corner(m: np.ndarray, room: np.ndarray, room_position: str) -> Tuple[int, int]:
    """Get the room offset for a corner placed room
    Args:
        m: The underlying map
        room: The room to insert 
        room_position: Position of the map the room is placed in

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


def _create_base_room(width: int, height: int, fill_tiles: Tuple[HiddenCellType] = rnd_background_tiles_hidden):
    m = np.zeros((height, width), dtype=np.uint8)
    for r in range(height):
        m[r, 0] = tilestr_to_hiddencellid["wall_brick"]  # left column
        m[r, width - 1] = tilestr_to_hiddencellid["wall_brick"]  # right column
    for c in range(width):
        m[0, c] = tilestr_to_hiddencellid["wall_brick"]  # top row
        m[height - 1, c] = tilestr_to_hiddencellid["wall_brick"]  # bottom row
    fill_room(m, fill_tiles)
    return m


def create_empty_room(room_width: int = 8, room_height: int = 8, fill_tiles: Tuple[HiddenCellType] = rnd_background_tiles_hidden):
    """Create an empty square map.
    Args:
        room_width: The width of the room
        room_height: The height of the room
        fill_tiles: Background tiles to fill room

    Returns:
        An empty room
    """
    m = _create_base_room(room_width, room_height, fill_tiles=fill_tiles)
    return m


def create_empty_map(map_size: int, fill_tiles: Tuple[HiddenCellType] = rnd_background_tiles_hidden):
    """Create an empty square map.
    Args:
        map_size: The width/height of the room
        fill_tiles: Background tiles to fill room

    Returns:
        An empty map
    """
    m = _create_base_room(map_size, map_size, fill_tiles=fill_tiles)
    return m
