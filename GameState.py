from Player import Player
from numpy import array, ndarray
from Piece import Piece
from Point import Point


class GameState:
    board: ndarray
    board_size: int

    current_player: Player
    waiting_player: Player

    undo_moves: list
    redo_moves: list

    all_moves: dict[Point, array]
    moves_with_capture: dict[Point, array]

    def __init__(self):
        """
        Initializes the object with the board.
        """

        W: Piece = Piece.WHITE_NORMAL
        R: Piece = Piece.RED_NORMAL
        E: Piece = Piece.EMPTY

        self.board = array([
            [E, W, E, W, E, W, E, W],
            [W, E, W, E, W, E, W, E],
            [E, W, E, W, E, W, E, W],
            [E, E, E, E, E, E, E, E],
            [E, E, E, E, E, E, E, E],
            [R, E, R, E, R, E, R, E],
            [E, R, E, R, E, R, E, R],
            [R, E, R, E, R, E, R, E]
        ])

        self.board_size = len(self.board)

        self.undo_moves = list()
        self.redo_moves = list()

    def reset(self):
        """
        Reinitializes the board.
        :return: None.
        """

        self.__init__()
