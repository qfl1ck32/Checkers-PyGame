from PlayerType import PlayerType
from Piece import Piece


class Player:
    name: str
    type: PlayerType
    normal_piece: Piece
    king_piece: Piece

    def __init__(self, name: str, normal_piece: Piece, king_piece: Piece, type: PlayerType):
        self.name = name
        self.normal_piece = normal_piece
        self.king_piece = king_piece
        self.type = type