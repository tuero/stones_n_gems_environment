import os
import sys
import numpy as np
import cv2

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from util.rnd_definitions import rnd_tile_images_visible


def insert_tile(m: np.ndarray, r: int, c: int, tile_idx: int, tile_size: int):
    """Insert a tile into an image.

    Args:
        m: The image representation as an numpy array
        r: Row to insert
        c: col to insert
        tile_idx: Visible cell type to index the image map
        tile_size: Size the tile image should reshape to
    """
    r_s, c_s = r * tile_size, c * tile_size
    r_e, c_e = (r + 1) * tile_size, (c + 1) * tile_size
    m[r_s:r_e, c_s:c_e] = cv2.resize(
        rnd_tile_images_visible[tile_idx], dsize=(tile_size, tile_size), interpolation=cv2.INTER_NEAREST
    )


def rnd_state_to_img(state: np.ndarray, tile_size: int = 32):
    """Convert an RND observation from the environment to an image.

    Args:
        state: numpy array representing state of visible cell types
        tile_size: Size of the tiles for the image

    Returns:
        Numpy array representing image (H, W, C)
    """
    rows, cols = state.shape[1], state.shape[2]
    map_img = np.zeros((rows * tile_size, cols * tile_size, 3), dtype=np.uint8)
    for tile_id, layer in enumerate(state):
        for r in range(rows):
            for c in range(cols):
                if layer[r, c]:
                    insert_tile(map_img, r, c, tile_id, tile_size)
    return map_img
