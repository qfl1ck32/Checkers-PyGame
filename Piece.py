from enum import Enum


class Piece(Enum):
    WHITE_NORMAL = 0
    WHITE_KING = 1

    RED_NORMAL = 2
    RED_KING = 4

    WHITE_NORMAL_SELECTED = 8
    WHITE_KING_SELECTED = 16

    RED_NORMAL_SELECTED = 32
    RED_KING_SELECTED = 64

    EMPTY = 128