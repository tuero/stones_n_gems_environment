import os
from enum import IntEnum
import cv2
import numpy as np


# Actions with corresponding offsets for the environment
ActionOffsets = {
    0: (0, 0),
    1: (-1, 0),
    2: (0, 1),
    3: (1, 0),
    4: (0, -1),
}


# Property bit flags
class RewardCodes(IntEnum):
    kRewardAgentDies = 1 << 0
    kRewardCollectDiamond = 1 << 1
    kRewardWalkThroughExit = 1 << 2
    kRewardNutToDiamond = 1 << 3
    kRewardCollectKey = 1 << 4
    kRewardWalkThroughGate = 1 << 5
    kRewardButterflyToDiamond = 1 << 6


# Hidden celltypes for the OpenSpiel stones_and_gems environment.
class HiddenCellType(IntEnum):
    kNull = -1
    kAgent = 0
    kEmpty = 1
    kDirt = 2
    kStone = 3
    kStoneFalling = 4
    kDiamond = 5
    kDiamondFalling = 6
    kExitClosed = 7
    kExitOpen = 8
    kAgentInExit = 9
    kFireflyUp = 10
    kFireflyLeft = 11
    kFireflyDown = 12
    kFireflyRight = 13
    kButterflyUp = 14
    kButterflyLeft = 15
    kButterflyDown = 16
    kButterflyRight = 17
    kWallBrick = 18
    kWallSteel = 19
    kWallMagicDormant = 20
    kWallMagicOn = 21
    kWallMagicExpired = 22
    kBlob = 23
    kExplosionDiamond = 24
    kExplosionBoulder = 25
    kExplosionEmpty = 26
    kGateRedClosed = 27
    kGateRedOpen = 28
    kKeyRed = 29
    kGateBlueClosed = 30
    kGateBlueOpen = 31
    kKeyBlue = 32
    kGateGreenClosed = 33
    kGateGreenOpen = 34
    kKeyGreen = 35
    kGateYellowClosed = 36
    kGateYellowOpen = 37
    kKeyYellow = 38
    kNut = 39
    kNutFalling = 40
    kBomb = 41
    kBombFalling = 42
    kOrangeUp = 43
    kOrangeLeft = 44
    kOrangeDown = 45
    kOrangeRight = 46
    kPebbleInDirt = 47
    kStoneInDirt = 48
    kVoidInDirt = 49


# Total number of hidden cell types
NUM_HIDDEN_CELL_TYPE = 50

# Visible celltypes for the OpenSpiel stones_and_gems environment.
class VisibleCellType(IntEnum):
    kNull = -1
    kAgent = 0
    kEmpty = 1
    kDirt = 2
    kStone = 3
    kDiamond = 4
    kExitClosed = 5
    kExitOpen = 6
    kAgentInExit = 7
    kFirefly = 8
    kButterfly = 9
    kWallBrick = 10
    kWallSteel = 11
    kWallMagicOff = 12
    kWallMagicOn = 13
    kBlob = 14
    kExplosion = 15
    kGateRedClosed = 16
    kGateRedOpen = 17
    kKeyRed = 18
    kGateBlueClosed = 19
    kGateBlueOpen = 20
    kKeyBlue = 21
    kGateGreenClosed = 22
    kGateGreenOpen = 23
    kKeyGreen = 24
    kGateYellowClosed = 25
    kGateYellowOpen = 26
    kKeyYellow = 27
    kNut = 28
    kBomb = 29
    kOrange = 30
    kPebbleInDirt = 31
    kStoneInDirt = 32
    kVoidInDirt = 33


# Total number of visible cell types
NUM_VISIBLE_CELL_TYPE = 34


HiddenToVisibleMapping = {
    HiddenCellType.kNull: VisibleCellType.kNull,
    HiddenCellType.kAgent: VisibleCellType.kAgent,
    HiddenCellType.kEmpty: VisibleCellType.kEmpty,
    HiddenCellType.kDirt: VisibleCellType.kDirt,
    HiddenCellType.kStone: VisibleCellType.kStone,
    HiddenCellType.kStoneFalling: VisibleCellType.kStone,
    HiddenCellType.kDiamond: VisibleCellType.kDiamond,
    HiddenCellType.kDiamondFalling: VisibleCellType.kDiamond,
    HiddenCellType.kExitClosed: VisibleCellType.kExitClosed,
    HiddenCellType.kExitOpen: VisibleCellType.kExitOpen,
    HiddenCellType.kAgentInExit: VisibleCellType.kAgentInExit,
    HiddenCellType.kFireflyUp: VisibleCellType.kFirefly,
    HiddenCellType.kFireflyLeft: VisibleCellType.kFirefly,
    HiddenCellType.kFireflyDown: VisibleCellType.kFirefly,
    HiddenCellType.kFireflyRight: VisibleCellType.kFirefly,
    HiddenCellType.kButterflyUp: VisibleCellType.kButterfly,
    HiddenCellType.kButterflyLeft: VisibleCellType.kButterfly,
    HiddenCellType.kButterflyDown: VisibleCellType.kButterfly,
    HiddenCellType.kButterflyRight: VisibleCellType.kButterfly,
    HiddenCellType.kWallBrick: VisibleCellType.kWallBrick,
    HiddenCellType.kWallSteel: VisibleCellType.kWallSteel,
    HiddenCellType.kWallMagicDormant: VisibleCellType.kWallMagicOff,
    HiddenCellType.kWallMagicOn: VisibleCellType.kWallMagicOn,
    HiddenCellType.kWallMagicExpired: VisibleCellType.kWallMagicOff,
    HiddenCellType.kBlob: VisibleCellType.kBlob,
    HiddenCellType.kExplosionDiamond: VisibleCellType.kExplosion,
    HiddenCellType.kExplosionBoulder: VisibleCellType.kExplosion,
    HiddenCellType.kExplosionEmpty: VisibleCellType.kExplosion,
    HiddenCellType.kGateRedClosed: VisibleCellType.kGateRedClosed,
    HiddenCellType.kGateRedOpen: VisibleCellType.kGateRedOpen,
    HiddenCellType.kKeyRed: VisibleCellType.kKeyRed,
    HiddenCellType.kGateBlueClosed: VisibleCellType.kGateBlueClosed,
    HiddenCellType.kGateBlueOpen: VisibleCellType.kGateBlueOpen,
    HiddenCellType.kKeyBlue: VisibleCellType.kKeyBlue,
    HiddenCellType.kGateGreenClosed: VisibleCellType.kGateGreenClosed,
    HiddenCellType.kGateGreenOpen: VisibleCellType.kGateGreenOpen,
    HiddenCellType.kKeyGreen: VisibleCellType.kKeyGreen,
    HiddenCellType.kGateYellowClosed: VisibleCellType.kGateYellowClosed,
    HiddenCellType.kGateYellowOpen: VisibleCellType.kGateYellowOpen,
    HiddenCellType.kKeyYellow: VisibleCellType.kKeyYellow,
    HiddenCellType.kNut: VisibleCellType.kNut,
    HiddenCellType.kNutFalling: VisibleCellType.kNut,
    HiddenCellType.kBomb: VisibleCellType.kBomb,
    HiddenCellType.kBombFalling: VisibleCellType.kBomb,
    HiddenCellType.kOrangeUp: VisibleCellType.kOrange,
    HiddenCellType.kOrangeLeft: VisibleCellType.kOrange,
    HiddenCellType.kOrangeDown: VisibleCellType.kOrange,
    HiddenCellType.kOrangeRight: VisibleCellType.kOrange,
    HiddenCellType.kPebbleInDirt: VisibleCellType.kPebbleInDirt,
    HiddenCellType.kStoneInDirt: VisibleCellType.kStoneInDirt,
    HiddenCellType.kVoidInDirt: VisibleCellType.kVoidInDirt,
}


# Map for visible tile id's to string names
visiblecelltype_to_str = {
    VisibleCellType.kAgent: "agent",
    VisibleCellType.kEmpty: "empty",
    VisibleCellType.kDirt: "dirt",
    VisibleCellType.kStone: "stone",
    VisibleCellType.kDiamond: "diamond",
    VisibleCellType.kExitClosed: "exit_closed",
    VisibleCellType.kExitOpen: "exit_open",
    VisibleCellType.kAgentInExit: "agent_in_exit",
    VisibleCellType.kFirefly: "firefly",
    VisibleCellType.kButterfly: "butterfly",
    VisibleCellType.kWallBrick: "wall_brick",
    VisibleCellType.kWallSteel: "wall_steel",
    VisibleCellType.kWallMagicOff: "wall_magic_off",
    VisibleCellType.kWallMagicOn: "wall_magic_on",
    VisibleCellType.kBlob: "blob",
    VisibleCellType.kExplosion: "explosion",
    VisibleCellType.kGateRedClosed: "gate_red_closed",
    VisibleCellType.kGateRedOpen: "gate_red_open",
    VisibleCellType.kKeyRed: "key_red",
    VisibleCellType.kGateBlueClosed: "gate_blue_closed",
    VisibleCellType.kGateBlueOpen: "gate_blue_open",
    VisibleCellType.kKeyBlue: "key_blue",
    VisibleCellType.kGateGreenClosed: "gate_green_closed",
    VisibleCellType.kGateGreenOpen: "gate_green_open",
    VisibleCellType.kKeyGreen: "key_green",
    VisibleCellType.kGateYellowClosed: "gate_yellow_closed",
    VisibleCellType.kGateYellowOpen: "gate_yellow_open",
    VisibleCellType.kKeyYellow: "key_yellow",
    VisibleCellType.kNut: "nut",
    VisibleCellType.kBomb: "bomb",
    VisibleCellType.kOrange: "orange",
    VisibleCellType.kPebbleInDirt: "pebble_in_dirt",
    VisibleCellType.kStoneInDirt: "stone_in_dirt",
    VisibleCellType.kVoidInDirt: "void_in_dirt",
}


# Map for hidden tile id's to string names
hiddencelltype_to_str = {
    HiddenCellType.kAgent: "agent",
    HiddenCellType.kEmpty: "empty",
    HiddenCellType.kDirt: "dirt",
    HiddenCellType.kStone: "stone",
    HiddenCellType.kStoneFalling: "stone",
    HiddenCellType.kDiamond: "diamond",
    HiddenCellType.kDiamondFalling: "diamond",
    HiddenCellType.kExitClosed: "exit_closed",
    HiddenCellType.kExitOpen: "exit_open",
    HiddenCellType.kAgentInExit: "agent_in_exit",
    HiddenCellType.kFireflyUp: "firefly",
    HiddenCellType.kFireflyLeft: "firefly",
    HiddenCellType.kFireflyDown: "firefly",
    HiddenCellType.kFireflyRight: "firefly",
    HiddenCellType.kButterflyUp: "butterfly",
    HiddenCellType.kButterflyLeft: "butterfly",
    HiddenCellType.kButterflyDown: "butterfly",
    HiddenCellType.kButterflyRight: "butterfly",
    HiddenCellType.kWallBrick: "wall_brick",
    HiddenCellType.kWallSteel: "wall_steel",
    HiddenCellType.kWallMagicDormant: "wall_magic_off",
    HiddenCellType.kWallMagicOn: "wall_magic_on",
    HiddenCellType.kWallMagicExpired: "wall_magic_off",
    HiddenCellType.kBlob: "blob",
    HiddenCellType.kExplosionDiamond: "explosion",
    HiddenCellType.kExplosionBoulder: "explosion",
    HiddenCellType.kExplosionEmpty: "explosion",
    HiddenCellType.kGateRedClosed: "gate_red_closed",
    HiddenCellType.kGateRedOpen: "gate_red_open",
    HiddenCellType.kKeyRed: "key_red",
    HiddenCellType.kGateBlueClosed: "gate_blue_closed",
    HiddenCellType.kGateBlueOpen: "gate_blue_open",
    HiddenCellType.kKeyBlue: "key_blue",
    HiddenCellType.kGateGreenClosed: "gate_green_closed",
    HiddenCellType.kGateGreenOpen: "gate_green_open",
    HiddenCellType.kKeyGreen: "key_green",
    HiddenCellType.kGateYellowClosed: "gate_yellow_closed",
    HiddenCellType.kGateYellowOpen: "gate_yellow_open",
    HiddenCellType.kKeyYellow: "key_yellow",
    HiddenCellType.kNut: "nut",
    HiddenCellType.kNutFalling: "nut",
    HiddenCellType.kBomb: "bomb",
    HiddenCellType.kBombFalling: "bomb",
    HiddenCellType.kOrangeUp: "orange",
    HiddenCellType.kOrangeLeft: "orange",
    HiddenCellType.kOrangeDown: "orange",
    HiddenCellType.kOrangeRight: "orange",
    HiddenCellType.kPebbleInDirt: "pebble_in_dirt",
    HiddenCellType.kStoneInDirt: "stone_in_dirt",
    HiddenCellType.kVoidInDirt: "void_in_dirt",
}


# Map for tile string name to ids
tilestr_to_visiblecellid = {v: k for k, v in visiblecelltype_to_str.items()}
tilestr_to_hiddencellid = {v: k for k, v in hiddencelltype_to_str.items()}


# Item groups
rnd_doors_visible = [
    tilestr_to_visiblecellid[name] for name in ["gate_red_closed", "gate_yellow_closed", "gate_green_closed", "gate_blue_closed"]
]
rnd_doorsopen_visible = [
    tilestr_to_visiblecellid[name] for name in ["gate_red_open", "gate_yellow_open", "gate_green_open", "gate_blue_open"]
]
rnd_keys_visible = [tilestr_to_visiblecellid[name] for name in ["key_red", "key_yellow", "key_green", "key_blue"]]
rnd_background_tiles_visible = [tilestr_to_visiblecellid[name] for name in ["empty", "dirt"]]
rnd_static_tiles_visible = [
    tilestr_to_visiblecellid[name] for name in ["empty", "dirt", "pebble_in_dirt", "stone_in_dirt", "void_in_dirt"]
]

rnd_doors_hidden = [
    tilestr_to_hiddencellid[name] for name in ["gate_red_closed", "gate_yellow_closed", "gate_green_closed", "gate_blue_closed"]
]
rnd_doorsopen_hidden = [
    tilestr_to_hiddencellid[name] for name in ["gate_red_open", "gate_yellow_open", "gate_green_open", "gate_blue_open"]
]
rnd_keys_hidden = [tilestr_to_hiddencellid[name] for name in ["key_red", "key_yellow", "key_green", "key_blue"]]
rnd_background_tiles_hidden = [tilestr_to_hiddencellid[name] for name in ["empty", "dirt"]]
rnd_static_tiles_hidden = [
    tilestr_to_hiddencellid[name] for name in ["empty", "dirt", "pebble_in_dirt", "stone_in_dirt", "void_in_dirt"]
]


# Images for the tiles
_asset_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "rnd_tiles/")
rnd_tile_images_visible = {
    k: cv2.cvtColor(cv2.imread(str(_asset_path) + v + ".png"), cv2.COLOR_BGR2RGB) for k, v in visiblecelltype_to_str.items()
}
rnd_tile_images_hidden = {
    k: cv2.cvtColor(cv2.imread(str(_asset_path) + v + ".png"), cv2.COLOR_BGR2RGB) for k, v in hiddencelltype_to_str.items()
}


# Tiles to cost maps
tile_costs_visible = {
    VisibleCellType.kAgent: 0,
    VisibleCellType.kEmpty: 0.1,
    VisibleCellType.kDirt: 3,
    VisibleCellType.kStone: 0,
    VisibleCellType.kDiamond: 0,
    VisibleCellType.kExitClosed: 60,
    VisibleCellType.kExitOpen: 1,
    VisibleCellType.kAgentInExit: 0,
    VisibleCellType.kFirefly: 0,
    VisibleCellType.kButterfly: 0,
    VisibleCellType.kWallBrick: 60,
    VisibleCellType.kWallSteel: 60,
    VisibleCellType.kWallMagicOff: 60,
    VisibleCellType.kWallMagicOn: 60,
    VisibleCellType.kBlob: 0,
    VisibleCellType.kExplosion: 0,
    VisibleCellType.kGateRedClosed: 15,
    VisibleCellType.kGateRedOpen: 1,
    VisibleCellType.kKeyRed: 1,
    VisibleCellType.kGateBlueClosed: 15,
    VisibleCellType.kGateBlueOpen: 1,
    VisibleCellType.kKeyBlue: 1,
    VisibleCellType.kGateGreenClosed: 15,
    VisibleCellType.kGateGreenOpen: 1,
    VisibleCellType.kKeyGreen: 1,
    VisibleCellType.kGateYellowClosed: 15,
    VisibleCellType.kGateYellowOpen: 1,
    VisibleCellType.kKeyYellow: 1,
    VisibleCellType.kNut: 0,
    VisibleCellType.kBomb: 0,
    VisibleCellType.kOrange: 0,
    VisibleCellType.kPebbleInDirt: 8,
    VisibleCellType.kStoneInDirt: 12,
    VisibleCellType.kVoidInDirt: 2,
}
tile_costs_hidden = {
    HiddenCellType.kAgent: 0,
    HiddenCellType.kEmpty: 0.1,
    HiddenCellType.kDirt: 3,
    HiddenCellType.kStone: 0,
    HiddenCellType.kStoneFalling: 0,
    HiddenCellType.kDiamond: 0,
    HiddenCellType.kDiamondFalling: 0,
    HiddenCellType.kExitClosed: 60,
    HiddenCellType.kExitOpen: 1,
    HiddenCellType.kAgentInExit: 0,
    HiddenCellType.kFireflyUp: 0,
    HiddenCellType.kFireflyLeft: 0,
    HiddenCellType.kFireflyDown: 0,
    HiddenCellType.kFireflyRight: 0,
    HiddenCellType.kButterflyUp: 0,
    HiddenCellType.kButterflyLeft: 0,
    HiddenCellType.kButterflyDown: 0,
    HiddenCellType.kButterflyRight: 0,
    HiddenCellType.kWallBrick: 60,
    HiddenCellType.kWallSteel: 60,
    HiddenCellType.kWallMagicDormant: 60,
    HiddenCellType.kWallMagicOn: 60,
    HiddenCellType.kWallMagicExpired: 60,
    HiddenCellType.kBlob: 0,
    HiddenCellType.kExplosionDiamond: 0,
    HiddenCellType.kExplosionBoulder: 0,
    HiddenCellType.kExplosionEmpty: 0,
    HiddenCellType.kGateRedClosed: 15,
    HiddenCellType.kGateRedOpen: 1,
    HiddenCellType.kKeyRed: 1,
    HiddenCellType.kGateBlueClosed: 15,
    HiddenCellType.kGateBlueOpen: 1,
    HiddenCellType.kKeyBlue: 1,
    HiddenCellType.kGateGreenClosed: 15,
    HiddenCellType.kGateGreenOpen: 1,
    HiddenCellType.kKeyGreen: 1,
    HiddenCellType.kGateYellowClosed: 15,
    HiddenCellType.kGateYellowOpen: 1,
    HiddenCellType.kKeyYellow: 1,
    HiddenCellType.kNut: 0,
    HiddenCellType.kNutFalling: 0,
    HiddenCellType.kBomb: 0,
    HiddenCellType.kBombFalling: 0,
    HiddenCellType.kOrangeUp: 0,
    HiddenCellType.kOrangeLeft: 0,
    HiddenCellType.kOrangeDown: 0,
    HiddenCellType.kOrangeRight: 0,
    HiddenCellType.kPebbleInDirt: 8,
    HiddenCellType.kStoneInDirt: 12,
    HiddenCellType.kVoidInDirt: 2,
}

# --------------
# MISC
# --------------
hlp_ids = {
    tilestr_to_hiddencellid["exit_open"]: 0,
    tilestr_to_hiddencellid["key_red"]: 1,
    tilestr_to_hiddencellid["gate_red_closed"]: 2,
    tilestr_to_hiddencellid["key_yellow"]: 3,
    tilestr_to_hiddencellid["gate_yellow_closed"]: 4,
    tilestr_to_hiddencellid["key_green"]: 5,
    tilestr_to_hiddencellid["gate_green_closed"]: 6,
    tilestr_to_hiddencellid["key_blue"]: 7,
    tilestr_to_hiddencellid["gate_blue_closed"]: 8,
}


def hiddencell_to_mapstr(map_ids: np.ndarray, max_steps: int, num_diamonds: int = 0):
    """Convert an array of HiddenCellTypes to string representation for OpenSpiel environment.

    Args:
        map_ids: np.array of HiddenCellTypes
        max_steps: Maximum number of steps for the environment before stopping
        num_diamonds: Number of diamonds required to open the door

    Returns:
        string representing map in OpenSpiel format
    """
    rows, cols = map_ids.shape
    map_str = "{},{},{},{}\n".format(rows, cols, max_steps, num_diamonds)
    for r in range(rows):
        for c in range(cols):
            map_str += "{},".format(map_ids[r, c])
        map_str = map_str[:-1] + "\n"
    return map_str[:-1]


def create_path_key(start_cell, goal_cell, key_red, door_red, key_yellow, door_yellow):
    return {
        -1: start_cell,
        0: goal_cell,
        1: key_red,
        2: door_red,
        3: key_yellow,
        4: door_yellow,
    }
