import sys
import os
from enum import IntEnum
from copy import deepcopy
from typing import Tuple, Dict
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from util.rnd_definitions import *


class Directions(IntEnum):
    kNone = 0
    kUp = 1
    kRight = 2
    kDown = 3
    kLeft = 4
    kUpRight = 5
    kDownRight = 6
    kDownLeft = 7
    kUpLeft = 8


NUM_DIRECTIONS = 9


class Actions(IntEnum):
    kNone = 0
    kUp = 1
    kRight = 2
    kDown = 3
    kLeft = 4


NUM_ACTIONS = 5


class ElementProperties(IntEnum):
    kNone = 0
    kConsumable = 1 << 0
    kCanExplode = 1 << 1
    kRounded = 1 << 2
    kTraversable = 1 << 3


# Element properties
ElementPropertiesMapping = {
    HiddenCellType.kNull: ElementProperties.kNone,
    HiddenCellType.kAgent: ElementProperties.kConsumable | ElementProperties.kCanExplode,
    HiddenCellType.kEmpty: ElementProperties.kConsumable | ElementProperties.kTraversable,
    HiddenCellType.kDirt: ElementProperties.kConsumable | ElementProperties.kTraversable,
    HiddenCellType.kStone: ElementProperties.kConsumable | ElementProperties.kRounded,
    HiddenCellType.kStoneFalling: ElementProperties.kConsumable,
    HiddenCellType.kDiamond: ElementProperties.kConsumable | ElementProperties.kRounded | ElementProperties.kTraversable,
    HiddenCellType.kDiamondFalling: ElementProperties.kConsumable,
    HiddenCellType.kExitClosed: ElementProperties.kNone,
    HiddenCellType.kExitOpen: ElementProperties.kTraversable,
    HiddenCellType.kAgentInExit: ElementProperties.kNone,
    HiddenCellType.kFireflyUp: ElementProperties.kConsumable | ElementProperties.kCanExplode,
    HiddenCellType.kFireflyLeft: ElementProperties.kConsumable | ElementProperties.kCanExplode,
    HiddenCellType.kFireflyDown: ElementProperties.kConsumable | ElementProperties.kCanExplode,
    HiddenCellType.kFireflyRight: ElementProperties.kConsumable | ElementProperties.kCanExplode,
    HiddenCellType.kButterflyUp: ElementProperties.kConsumable | ElementProperties.kCanExplode,
    HiddenCellType.kButterflyLeft: ElementProperties.kConsumable | ElementProperties.kCanExplode,
    HiddenCellType.kButterflyDown: ElementProperties.kConsumable | ElementProperties.kCanExplode,
    HiddenCellType.kButterflyRight: ElementProperties.kConsumable | ElementProperties.kCanExplode,
    HiddenCellType.kWallBrick: ElementProperties.kConsumable | ElementProperties.kRounded,
    HiddenCellType.kWallSteel: ElementProperties.kNone,
    HiddenCellType.kWallMagicDormant: ElementProperties.kConsumable,
    HiddenCellType.kWallMagicOn: ElementProperties.kConsumable,
    HiddenCellType.kWallMagicExpired: ElementProperties.kConsumable,
    HiddenCellType.kBlob: ElementProperties.kConsumable,
    HiddenCellType.kExplosionDiamond: ElementProperties.kNone,
    HiddenCellType.kExplosionBoulder: ElementProperties.kNone,
    HiddenCellType.kExplosionEmpty: ElementProperties.kNone,
    HiddenCellType.kGateRedClosed: ElementProperties.kNone,
    HiddenCellType.kGateRedOpen: ElementProperties.kNone,
    HiddenCellType.kKeyRed: ElementProperties.kTraversable,
    HiddenCellType.kGateBlueClosed: ElementProperties.kNone,
    HiddenCellType.kGateBlueOpen: ElementProperties.kNone,
    HiddenCellType.kKeyBlue: ElementProperties.kTraversable,
    HiddenCellType.kGateGreenClosed: ElementProperties.kNone,
    HiddenCellType.kGateGreenOpen: ElementProperties.kNone,
    HiddenCellType.kKeyGreen: ElementProperties.kTraversable,
    HiddenCellType.kGateYellowClosed: ElementProperties.kNone,
    HiddenCellType.kGateYellowOpen: ElementProperties.kNone,
    HiddenCellType.kKeyYellow: ElementProperties.kTraversable,
    HiddenCellType.kNut: ElementProperties.kRounded | ElementProperties.kConsumable,
    HiddenCellType.kNutFalling: ElementProperties.kRounded | ElementProperties.kConsumable,
    HiddenCellType.kBomb: ElementProperties.kRounded | ElementProperties.kConsumable | ElementProperties.kCanExplode,
    HiddenCellType.kBombFalling: ElementProperties.kRounded | ElementProperties.kConsumable | ElementProperties.kCanExplode,
    HiddenCellType.kOrangeUp: ElementProperties.kConsumable | ElementProperties.kCanExplode,
    HiddenCellType.kOrangeLeft: ElementProperties.kConsumable | ElementProperties.kCanExplode,
    HiddenCellType.kOrangeDown: ElementProperties.kConsumable | ElementProperties.kCanExplode,
    HiddenCellType.kOrangeRight: ElementProperties.kConsumable | ElementProperties.kCanExplode,
    HiddenCellType.kPebbleInDirt: ElementProperties.kConsumable,
    HiddenCellType.kStoneInDirt: ElementProperties.kConsumable,
    HiddenCellType.kVoidInDirt: ElementProperties.kConsumable,
}


# Element object definition
class Element:
    def __init__(
        self, cell_type: HiddenCellType, visible_type: VisibleCellType, properties: int, el_id: int = 0, has_updated: bool = False
    ):
        self.cell_type = cell_type
        self.visible_type = visible_type
        self.properties = properties
        self.el_id = el_id
        self.has_updated = has_updated

    def __eq__(self, other):
        return self.cell_type == other.cell_type

    def __ne__(self, other):
        return self.cell_type != other.cell_type

    def __hash__(self):
        return self.cell_type


# Factory to create elements from HiddenCellType
def element_factory(cell_type: HiddenCellType, el_id: int = 0) -> Element:
    return Element(
        cell_type=cell_type,
        visible_type=HiddenToVisibleMapping[cell_type],
        properties=ElementPropertiesMapping[cell_type],
        el_id=el_id,
    )


# All possible elements
kNullElement = element_factory(HiddenCellType.kNull)
kElAgent = element_factory(HiddenCellType.kAgent)
kElEmpty = element_factory(HiddenCellType.kEmpty)
kElDirt = element_factory(HiddenCellType.kDirt)
kElStone = element_factory(HiddenCellType.kStone)
kElStoneFalling = element_factory(HiddenCellType.kStoneFalling)
kElDiamond = element_factory(HiddenCellType.kDiamond)
kElDiamondFalling = element_factory(HiddenCellType.kDiamondFalling)
kElExitClosed = element_factory(HiddenCellType.kExitClosed)
kElExitOpen = element_factory(HiddenCellType.kExitOpen)
kElAgentInExit = element_factory(HiddenCellType.kAgentInExit)
kElFireflyUp = element_factory(HiddenCellType.kFireflyUp)
kElFireflyLeft = element_factory(HiddenCellType.kFireflyLeft)
kElFireflyDown = element_factory(HiddenCellType.kFireflyDown)
kElFireflyRight = element_factory(HiddenCellType.kFireflyRight)
kElButterflyUp = element_factory(HiddenCellType.kButterflyUp)
kElButterflyLeft = element_factory(HiddenCellType.kButterflyLeft)
kElButterflyDown = element_factory(HiddenCellType.kButterflyDown)
kElButterflyRight = element_factory(HiddenCellType.kButterflyRight)
kElWallBrick = element_factory(HiddenCellType.kWallBrick)
kElWallSteel = element_factory(HiddenCellType.kWallSteel)
kElWallMagicDormant = element_factory(HiddenCellType.kWallMagicDormant)
kElWallMagicOn = element_factory(HiddenCellType.kWallMagicOn)
kElWallMagicExpired = element_factory(HiddenCellType.kWallMagicExpired)
kElBlob = element_factory(HiddenCellType.kBlob)
kElExplosionDiamond = element_factory(HiddenCellType.kExplosionDiamond)
kElExplosionBoulder = element_factory(HiddenCellType.kExplosionBoulder)
kElExplosionEmpty = element_factory(HiddenCellType.kExplosionEmpty)
kElGateRedClosed = element_factory(HiddenCellType.kGateRedClosed)
kElGateRedOpen = element_factory(HiddenCellType.kGateRedOpen)
kElKeyRed = element_factory(HiddenCellType.kKeyRed)
kElGateBlueClosed = element_factory(HiddenCellType.kGateBlueClosed)
kElGateBlueOpen = element_factory(HiddenCellType.kGateBlueOpen)
kElKeyBlue = element_factory(HiddenCellType.kKeyBlue)
kElGateGreenClosed = element_factory(HiddenCellType.kGateGreenClosed)
kElGateGreenOpen = element_factory(HiddenCellType.kGateGreenOpen)
kElKeyGreen = element_factory(HiddenCellType.kKeyGreen)
kElGateYellowClosed = element_factory(HiddenCellType.kGateYellowClosed)
kElGateYellowOpen = element_factory(HiddenCellType.kGateYellowOpen)
kElKeyYellow = element_factory(HiddenCellType.kKeyYellow)
kElNut = element_factory(HiddenCellType.kNut)
kElNutFalling = element_factory(HiddenCellType.kNutFalling)
kElBomb = element_factory(HiddenCellType.kBomb)
kElBombFalling = element_factory(HiddenCellType.kBombFalling)
kElOrangeUp = element_factory(HiddenCellType.kOrangeUp)
kElOrangeLeft = element_factory(HiddenCellType.kOrangeLeft)
kElOrangeDown = element_factory(HiddenCellType.kOrangeDown)
kElOrangeRight = element_factory(HiddenCellType.kOrangeRight)
kElPebbleInDirt = element_factory(HiddenCellType.kPebbleInDirt)
kElStoneInDirt = element_factory(HiddenCellType.kStoneInDirt)
kElVoidInDirt = element_factory(HiddenCellType.kVoidInDirt)


# Mapping for grid to Element
kHiddenCellTypeToElement = {
    HiddenCellType.kNull: kNullElement,
    HiddenCellType.kAgent: kElAgent,
    HiddenCellType.kEmpty: kElEmpty,
    HiddenCellType.kDirt: kElDirt,
    HiddenCellType.kStone: kElStone,
    HiddenCellType.kStoneFalling: kElStoneFalling,
    HiddenCellType.kDiamond: kElDiamond,
    HiddenCellType.kDiamondFalling: kElDiamondFalling,
    HiddenCellType.kExitClosed: kElExitClosed,
    HiddenCellType.kExitOpen: kElExitOpen,
    HiddenCellType.kAgentInExit: kElAgentInExit,
    HiddenCellType.kFireflyUp: kElFireflyUp,
    HiddenCellType.kFireflyLeft: kElFireflyLeft,
    HiddenCellType.kFireflyDown: kElFireflyDown,
    HiddenCellType.kFireflyRight: kElFireflyRight,
    HiddenCellType.kButterflyUp: kElButterflyUp,
    HiddenCellType.kButterflyLeft: kElButterflyLeft,
    HiddenCellType.kButterflyDown: kElButterflyDown,
    HiddenCellType.kButterflyRight: kElButterflyRight,
    HiddenCellType.kWallBrick: kElWallBrick,
    HiddenCellType.kWallSteel: kElWallSteel,
    HiddenCellType.kWallMagicDormant: kElWallMagicDormant,
    HiddenCellType.kWallMagicOn: kElWallMagicOn,
    HiddenCellType.kWallMagicExpired: kElWallMagicExpired,
    HiddenCellType.kBlob: kElBlob,
    HiddenCellType.kExplosionDiamond: kElExplosionDiamond,
    HiddenCellType.kExplosionBoulder: kElExplosionBoulder,
    HiddenCellType.kExplosionEmpty: kElExplosionEmpty,
    HiddenCellType.kGateRedClosed: kElGateRedClosed,
    HiddenCellType.kGateRedOpen: kElGateRedOpen,
    HiddenCellType.kKeyRed: kElKeyRed,
    HiddenCellType.kGateBlueClosed: kElGateBlueClosed,
    HiddenCellType.kGateBlueOpen: kElGateBlueOpen,
    HiddenCellType.kKeyBlue: kElKeyBlue,
    HiddenCellType.kGateGreenClosed: kElGateGreenClosed,
    HiddenCellType.kGateGreenOpen: kElGateGreenOpen,
    HiddenCellType.kKeyGreen: kElKeyGreen,
    HiddenCellType.kGateYellowClosed: kElGateYellowClosed,
    HiddenCellType.kGateYellowOpen: kElGateYellowOpen,
    HiddenCellType.kKeyYellow: kElKeyYellow,
    HiddenCellType.kNut: kElNut,
    HiddenCellType.kNutFalling: kElNutFalling,
    HiddenCellType.kBomb: kElBomb,
    HiddenCellType.kBombFalling: kElBombFalling,
    HiddenCellType.kOrangeUp: kElOrangeUp,
    HiddenCellType.kOrangeLeft: kElOrangeLeft,
    HiddenCellType.kOrangeDown: kElOrangeDown,
    HiddenCellType.kOrangeRight: kElOrangeRight,
    HiddenCellType.kPebbleInDirt: kElPebbleInDirt,
    HiddenCellType.kStoneInDirt: kElStoneInDirt,
    HiddenCellType.kVoidInDirt: kElVoidInDirt,
}  # type: Dict[HiddenCellType, Element]


# Rotate actions right
kRotateRight = {
    Directions.kUp: Directions.kRight,
    Directions.kRight: Directions.kDown,
    Directions.kDown: Directions.kLeft,
    Directions.kLeft: Directions.kUp,
    Directions.kNone: Directions.kNone,
}  # type: Dict[Directions, Directions]

# Rotate actions left
kRotateLeft = {
    Directions.kUp: Directions.kLeft,
    Directions.kLeft: Directions.kDown,
    Directions.kDown: Directions.kRight,
    Directions.kRight: Directions.kUp,
    Directions.kNone: Directions.kNone,
}  # type: Dict[Directions, Directions]

# directions to offsets (col : row)
kDirectionOffsets = {
    Directions.kUp: (0, -1),
    Directions.kUpLeft: (-1, -1),
    Directions.kLeft: (-1, 0),
    Directions.kDownLeft: (-1, 1),
    Directions.kDown: (0, 1),
    Directions.kDownRight: (1, 1),
    Directions.kRight: (1, 0),
    Directions.kUpRight: (1, -1),
    Directions.kNone: (0, 0),
}  # type: Dict[Directions, Tuple[int, int]]

# Directions to fireflys
kDirectionToFirefly = {
    Directions.kUp: kElFireflyUp,
    Directions.kLeft: kElFireflyLeft,
    Directions.kDown: kElFireflyDown,
    Directions.kRight: kElFireflyRight,
}  # type: Dict[Directions, Element]

# Firefly to directions
kFireflyToDirection = {
    kElFireflyUp: Directions.kUp,
    kElFireflyLeft: Directions.kLeft,
    kElFireflyDown: Directions.kDown,
    kElFireflyRight: Directions.kRight,
}  # type: Dict[Element, Directions]

# Directions to butterflys
kDirectionToButterfly = {
    Directions.kUp: kElButterflyUp,
    Directions.kLeft: kElButterflyLeft,
    Directions.kDown: kElButterflyDown,
    Directions.kRight: kElButterflyRight,
}  # type: Dict[Directions, Element]

# Butterfly to directions
kButterflyToDirection = {
    kElButterflyUp: Directions.kUp,
    kElButterflyLeft: Directions.kLeft,
    kElButterflyDown: Directions.kDown,
    kElButterflyRight: Directions.kRight,
}  # type: Dict[Element, Directions]

# Orange to directions
kOrangeToDirection = {
    kElOrangeUp: Directions.kUp,
    kElOrangeLeft: Directions.kLeft,
    kElOrangeDown: Directions.kDown,
    kElOrangeRight: Directions.kRight,
}  # type: Dict[Element, Directions]

# Direction to Orange
kDirectionToOrange = {
    Directions.kUp: kElOrangeUp,
    Directions.kLeft: kElOrangeLeft,
    Directions.kDown: kElOrangeDown,
    Directions.kRight: kElOrangeRight,
}  # type: Dict[Directions, Element]

# Element explosion maps
kElementToExplosion = {
    kElFireflyUp: kElExplosionEmpty,
    kElFireflyLeft: kElExplosionEmpty,
    kElFireflyDown: kElExplosionEmpty,
    kElFireflyRight: kElExplosionEmpty,
    kElButterflyUp: kElExplosionDiamond,
    kElButterflyLeft: kElExplosionDiamond,
    kElButterflyDown: kElExplosionDiamond,
    kElButterflyRight: kElExplosionDiamond,
    kElAgent: kElExplosionEmpty,
    kElBomb: kElExplosionEmpty,
    kElBombFalling: kElExplosionEmpty,
    kElOrangeUp: kElExplosionEmpty,
    kElOrangeLeft: kElExplosionEmpty,
    kElOrangeDown: kElExplosionEmpty,
    kElOrangeRight: kElExplosionEmpty,
}  # type: Dict[Element, Element]

# Explosions back to elements
kExplosionToElement = {
    kElExplosionDiamond: kElDiamond,
    kElExplosionBoulder: kElStone,
    kElExplosionEmpty: kElEmpty,
}  # type: Dict[Element, Element]

# Magic wall conversion map
kMagicWallConversion = {
    kElStoneFalling: kElDiamondFalling,
    kElDiamondFalling: kElStoneFalling,
}  # type: Dict[Element, Element]

# Gem point maps
kGemPoints = {
    kElDiamond: 10,
    kElDiamondFalling: 10,
    kElAgentInExit: 100,
}  # type: Dict[Element, int]

# Gate open conversion map
kGateOpenMap = {
    kElGateRedClosed: kElGateRedOpen,
    kElGateBlueClosed: kElGateBlueOpen,
    kElGateGreenClosed: kElGateGreenOpen,
    kElGateYellowClosed: kElGateYellowOpen,
}  # type: Dict[Element, Element]

# Gate key map
kKeyToGate = {
    kElKeyRed: kElGateRedClosed,
    kElKeyBlue: kElGateBlueClosed,
    kElKeyGreen: kElGateGreenClosed,
    kElKeyYellow: kElGateYellowClosed,
}  # type: Dict[Element, Element]

# Stationary to falling
kElToFalling = {
    kElDiamond: kElDiamondFalling,
    kElStone: kElStoneFalling,
    kElNut: kElNutFalling,
    kElBomb: kElBombFalling,
}  # type: Dict[Element, Element]

# Element helper functions
def IsActionHorz(action) -> bool:
    return action == Directions.kLeft or action == Directions.kRight


def IsFirefly(element: Element) -> bool:
    return element == kElFireflyUp or element == kElFireflyLeft or element == kElFireflyDown or element == kElFireflyRight


def IsButterfly(element: Element) -> bool:
    return element == kElButterflyUp or element == kElButterflyLeft or element == kElButterflyDown or element == kElButterflyRight


def IsOrange(element: Element) -> bool:
    return element == kElOrangeUp or element == kElOrangeLeft or element == kElOrangeDown or element == kElOrangeRight


def IsExplosion(element: Element) -> bool:
    return element == kElExplosionBoulder or element == kElExplosionDiamond or element == kElExplosionEmpty


def IsMagicWall(element: Element) -> bool:
    return element == kElWallMagicDormant or element == kElWallMagicExpired or element == kElWallMagicOn


def IsOpenGate(element: Element) -> bool:
    return element == kElGateRedOpen or element == kElGateBlueOpen or element == kElGateGreenOpen or element == kElGateYellowOpen


def IsKey(element: Element) -> bool:
    return element == kElKeyRed or element == kElKeyBlue or element == kElKeyGreen or element == kElKeyYellow


def coord_from_action(coord: Tuple[int, int], action: Directions) -> Tuple[int, int]:
    row, col = coord
    offset = kDirectionOffsets[action]
    return row + offset[1], col + offset[0]
