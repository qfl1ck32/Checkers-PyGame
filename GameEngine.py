from __future__ import annotations

import numpy as np
from Point import Point
from copy import copy, deepcopy
from GameState import GameState
from Player import Player
from Piece import Piece
from PlayerType import PlayerType

from random import randint


class GameEngine:
    _change_in_coordinates_forward: list[Point]
    _change_in_coordinates_backward: list[Point]
    _change_in_coordinates_king: list[Point]

    _player_1: Player
    _player_2: Player

    game_state: GameState

    use_alpha_beta: bool

    search_algorithm_number_of_nodes_each_turn: list
    current_nodes_count: int

    search_algorithm_depth: int
    flip: bool

    def __init__(self, game_state: GameState, use_alpha_beta: bool, search_algorithm_depth: int,
                 player_1_name: str, player_2_name: str, player_1_type: PlayerType, player_2_type: PlayerType,
                 flip: bool) -> None:

        """
        Initializes the object.
        :param game_state: A GameState instance.
        :param use_alpha_beta: A boolean value, representing whether or not to use Alpha-Beta
        as the searching algorithm for the AI.
        :param search_algorithm_depth: The depth for the search algorithm.
        :param player_1_name: Player 1's name.
        :param player_2_name: Player 2's name.
        :param player_1_type: Player 1's type.
        :param player_2_type: Player 2's type.
        :param flip: A boolean value, representing whether or not to flip the board (i.e. swap red pieces for white)
        """

        self._change_in_coordinates_forward = [Point(-1, 1), Point(-1, -1)]
        self._change_in_coordinates_backward = [Point(1, 1), Point(1, -1)]
        self._change_in_coordinates_king = self._change_in_coordinates_forward + self._change_in_coordinates_backward

        self._player_1 = Player(player_1_name, Piece.RED_NORMAL, Piece.RED_KING, player_1_type)
        self._player_2 = Player(player_2_name, Piece.WHITE_NORMAL, Piece.WHITE_KING, player_2_type)

        self.game_state = game_state

        self.game_state.current_player = self._player_1
        self.game_state.waiting_player = self._player_2

        self.flip = flip

        if self.flip:
            self.game_state.board = np.flip(self.game_state.board)
            self._change_in_coordinates_backward, self._change_in_coordinates_forward = \
                self._change_in_coordinates_forward, self._change_in_coordinates_backward

        self._generate_and_store_all_moves()

        self.use_alpha_beta = use_alpha_beta

        self.search_algorithm_number_of_nodes_each_turn = list()
        self.current_nodes_count = 0

        self.search_algorithm_depth = search_algorithm_depth

    def reset(self):
        """
        Resets the game state.
        :return: None.
        """

        self.game_state.reset()

        if self.flip:
            self.game_state.board = np.flip(self.game_state.board)

        self.game_state.current_player, self.game_state.waiting_player = self._player_1, self._player_2
        self._generate_and_store_all_moves()

    def _swap_moves(self, game_state: GameState = None):
        """
        Swaps the current and the waiting players.
        :param game_state: A GameState instance or None.
        :return: None.
        """

        if game_state is None:
            game_state = self.game_state

        game_state.current_player, game_state.waiting_player = \
            game_state.waiting_player, game_state.current_player

    def _check_valid_coordinates(self, row: int, column: int, game_state: GameState = None) -> bool:
        """
        Checks if given coordinates are valid on the board.
        :param row: The row to check.
        :param column: The column to check.
        :param game_state: A GameState instance or None.
        :return: A boolean value, representing whether or not the coordinates are valid.
        """

        if game_state is None:
            game_state = self.game_state

        return game_state.board_size > row >= 0 and game_state.board_size > column >= 0

    def _get_coordinates_change(self, row: int, column: int, game_state: GameState = None) -> list[Point]:
        """
        Gives a list of points, representing the changes in direction that a piece can make.
        :param row: The row of the piece.
        :param column: The column of the piece.
        :param game_state: A GameState instance or None.
        :return: A list of Points.
        """

        if game_state is None:
            game_state = self.game_state

        if game_state.board[row][column] in (Piece.WHITE_KING, Piece.RED_KING):
            return self._change_in_coordinates_king

        return self._change_in_coordinates_forward if game_state.current_player == self._player_1 else \
            self._change_in_coordinates_backward

    def _generate_all_moves(self, game_state: GameState = None) -> tuple[dict[Point, np.array], dict[Point, np.array]]:
        """
        Generates all the next possible moves.
        :param game_state: A GameState instance or None.
        :return: Two dictionaries of the form d[point_from] = [point_to_1, point_to_2, ...]
        """

        if game_state is None:
            game_state = self.game_state

        all_moves: dict = dict()
        moves_with_capture: dict = dict()

        rows: np.array
        columns: np.array

        rows, columns = np.where(
            (game_state.board == game_state.current_player.normal_piece) |
            (game_state.board == game_state.current_player.king_piece)
        )

        for row, column in zip(rows, columns):
            moves: np.array
            with_capture: np.array

            moves, with_capture = self._get_coordinates_valid_moves_and_can_capture(row=row,
                                                                                    column=column,
                                                                                    game_state=game_state)

            current_point: Point = Point(row, column)

            if np.size(moves):
                if current_point not in all_moves.keys():
                    all_moves[current_point] = list()
                all_moves[current_point] = np.append(all_moves[current_point], moves)

            if np.size(with_capture):
                if current_point not in moves_with_capture.keys():
                    moves_with_capture[current_point] = list()
                moves_with_capture[current_point] = np.append(moves_with_capture[current_point], with_capture)

        return all_moves, moves_with_capture

    def _generate_and_store_all_moves(self, game_state: GameState = None):
        """
        Generates all the next moves using _generate_all_moves() and stores them into the given game state.
        :param game_state: A GameState instance or None.
        :return: None.
        """

        if game_state is None:
            game_state = self.game_state

        game_state.all_moves, game_state.moves_with_capture = self._generate_all_moves(game_state=game_state)

    def _get_coordinates_valid_moves_and_can_capture(self, row: int, column: int, game_state: GameState = None) \
            -> tuple[np.array, np.array]:
        """
        Gives a tuple consisting of all the next possible moves and all the moves that capture a piece.
        :param row: The row of the piece to be moved.
        :param column: The column of the piece to be moved.
        :param game_state: A GameState instance or None.
        :return: A tuple of two numpy arrays.
        """

        if game_state is None:
            game_state = self.game_state

        all_moves: list = list()
        with_capture: list = list()

        for change in self._get_coordinates_change(row=row,
                                                   column=column,
                                                   game_state=game_state):

            can_capture: bool = False

            new_row: int = row + change.row
            new_column: int = column + change.column

            if not self._check_valid_coordinates(row=new_row,
                                                 column=new_column,
                                                 game_state=game_state):
                continue

            if game_state.board[new_row][new_column] == game_state.waiting_player.normal_piece or \
                    game_state.board[new_row][new_column] == game_state.waiting_player.king_piece and \
                    game_state.board[row][column] == game_state.current_player.king_piece:
                new_row -= 1 if row > new_row else -1
                new_column -= 1 if column > new_column else -1
                can_capture = True

            if not self._check_valid_coordinates(row=new_row,
                                                 column=new_column,
                                                 game_state=game_state) or \
                    game_state.board[new_row][new_column] != Piece.EMPTY:
                continue

            new_point: Point = Point(row=new_row,
                                     column=new_column)

            all_moves.append(new_point)

            if can_capture:
                with_capture.append(new_point)

        return np.array(all_moves), np.array(with_capture)

    def test_winner(self, game_state: GameState = None) -> bool:
        """
        Checks if the state of a game is final.
        :param game_state: A GameState instance or None.
        :return: A boolean value, representing whether or not the state of the game is final.
        """

        if game_state is None:
            game_state = self.game_state

        return not (
                np.size(np.where(
                    (game_state.board == self._player_1.normal_piece) | (
                            game_state.board == self._player_1.king_piece))) and
                np.size(
                    np.where(
                        (game_state.board == self._player_2.normal_piece) | (
                                game_state.board == self._player_2.king_piece))) and
                len(game_state.all_moves)
        )

    def undo(self, game_state: GameState = None) -> tuple[Point, Point, Point, Piece] or None:
        """
        Undoes the previous move.
        :param game_state: A GameState instance or None.
        :return: None if there are no previous moves, else a tuple of (point_from, point_to, piece_from, jumped_piece).
        """

        if game_state is None:
            game_state = self.game_state

        if not game_state.undo_moves or len(game_state.undo_moves) == 1 and \
                ((self._player_1.type == PlayerType.COMPUTER or self._player_2.type == PlayerType.COMPUTER) and
                 game_state.current_player.type != PlayerType.COMPUTER):
            return None

        point_from: Point
        point_to: Point
        piece_from: Piece
        jumped_piece: Point or None

        game_state.redo_moves.append(game_state.undo_moves[-1])

        point_from, point_to, piece_from, has_swapped_moves, jumped_piece = game_state.undo_moves.pop()

        row_from: int
        column_from: int

        row_to: int
        column_to: int

        row_from, column_from = point_from.row, point_from.column
        row_to, column_to = point_to.row, point_to.column

        game_state.board[row_from][column_from], game_state.board[row_to][column_to] = piece_from, Piece.EMPTY

        if jumped_piece is not None:
            game_state.board[(row_from + row_to) >> 1][(column_from + column_to) >> 1] = jumped_piece

        if has_swapped_moves:
            self._swap_moves(game_state=game_state)

        self._generate_and_store_all_moves(game_state=game_state)

        return point_from, point_to, piece_from, jumped_piece

    def redo(self, game_state: GameState = None) -> tuple[Point, Point, Point, Piece] or None:
        """
        Redoes the last action that was undoed.
        :param game_state: A GameState instance.
        :return: None if there was no undoed moves or a tuple of (point_from, point_to, piece_from, jumped_piece).
        """

        if game_state is None:
            game_state = self.game_state

        if not game_state.redo_moves:
            return None

        point_from: Point
        point_to: Point
        piece_from: Piece

        point_from, point_to, piece_from, has_swapped_moves, jumped_piece = game_state.redo_moves.pop()

        self.make_move(point_from=point_from,
                       point_to=point_to,
                       clear_redo=False)

        return point_from, point_to, piece_from, jumped_piece

    def get_coordinates_valid_moves(self, row: int, column: int, game_state: GameState = None) -> np.array:
        """
        Gives a list of all the coordinates that a piece can move to.
        :param row: The row of the piece to move.
        :param column: The column of the piece to move.
        :param game_state: A GameState instance.
        :return: A numpy array if there are possible moves or an empty list otherwise.
        """

        if game_state is None:
            game_state = self.game_state

        current_point: Point = Point(row=row,
                                     column=column)

        try:
            if len(game_state.moves_with_capture) and current_point not in game_state.moves_with_capture.keys():
                return []

            try:
                return game_state.moves_with_capture[current_point] if \
                    len(game_state.moves_with_capture[current_point]) else \
                    game_state.all_moves[current_point]
            except:
                return game_state.all_moves[current_point]
        except KeyError:
            return []

    def make_move(self, point_from: Point, point_to: Point, clear_redo: bool = True, game_state: GameState = None) \
            -> tuple[Piece, Piece]:
        """
        Makes a move.
        :param point_from: The point from which to move.
        :param point_to: The point to move at.
        :param clear_redo: A boolean value, representing whether or not to clear all the redo history.
        :param game_state: A GameState instance.
        :return: A tuple of (piece_from, jumped_piece).
        """

        if game_state is None:
            game_state = self.game_state

        row_from: int = point_from.row
        column_from: int = point_from.column

        row_to: int = point_to.row
        column_to: int = point_to.column

        piece_from: Piece = game_state.board[row_from][column_from]

        game_state.board[row_to][column_to], game_state.board[row_from][column_from] = piece_from, Piece.EMPTY

        jumped_piece: Piece = game_state.board[(row_to + row_from) >> 1][(column_to + column_from) >> 1]

        jumped: bool = False

        if jumped_piece in (game_state.waiting_player.normal_piece, game_state.waiting_player.king_piece):
            game_state.board[(row_to + row_from) >> 1][(column_to + column_from) >> 1] = Piece.EMPTY
            jumped = True

        if game_state.current_player == self._player_1:
            if not row_to or self.flip and row_to == game_state.board_size - 1:
                game_state.board[row_to][column_to] = self._player_1.king_piece
        else:
            if row_to == game_state.board_size - 1 or self.flip and not row_to:
                game_state.board[row_to][column_to] = self._player_2.king_piece

        next_moves: np.array
        next_moves_can_jump: np.array

        self._generate_and_store_all_moves(game_state=game_state)

        has_swapped_moves: bool = False

        if not (jumped and len(game_state.moves_with_capture)):
            self._swap_moves(game_state=game_state)
            has_swapped_moves = True

            self._generate_and_store_all_moves(game_state=game_state)

        game_state.undo_moves.append((copy(point_from), copy(point_to), piece_from, has_swapped_moves,
                                      jumped_piece if jumped else None))

        if clear_redo:
            game_state.redo_moves.clear()

        return piece_from, jumped_piece

    def make_best_move(self, game_state: GameState = None) -> tuple[Point, Point, Piece, Piece]:
        """
        Makes the best move of the given game state.
        :param game_state: A GameState instance.
        :return: A tuple of (point_from, point_to, piece_from, jumped_piece).
        """

        if game_state is None:
            game_state = self.game_state

        best_moves: list

        if self.use_alpha_beta:
            best_moves, _ = self.alpha_beta(game_state=game_state,
                                            depth=self.search_algorithm_depth,
                                            alpha=float('-inf'),
                                            beta=float('+inf'),
                                            maximizing_player=True)

        else:
            best_moves, _ = self.mini_max(game_state=game_state,
                                          depth=self.search_algorithm_depth,
                                          maximizing_player=True)

        self.search_algorithm_number_of_nodes_each_turn.append(self.current_nodes_count)
        self.current_nodes_count = 0

        (point_from, point_to) = best_moves[randint(0, len(best_moves) - 1)]

        piece_from, jumped_piece = self.make_move(point_from, point_to)

        return point_from, point_to, piece_from, jumped_piece

    def _evaluate_state(self, game_state: GameState = None, strong: bool = False) -> float:
        """
        Evaluates a state.

        The function is indeed a good estimator, because it returns a positive or negative value, based on
        the number of remaining pieces - if there are more pieces for the AI, the score will be positive.

        :param game_state: A GameState instance.
        :param strong: A boolean value, representing whether or not to use the strong version of the function
        (which gives more importance to king pieces).
        :return: A float value.
        """

        if game_state is None:
            game_state = self.game_state

        board: np.ndarray = game_state.board

        player_1_normal_pieces_count: int = np.size(np.where(board == game_state.current_player.normal_piece)) // 2
        player_2_normal_pieces_count: int = np.size(np.where(board == game_state.waiting_player.normal_piece)) // 2

        player_1_king_pieces_count: int = np.size(np.where(board == game_state.current_player.king_piece)) // 2
        player_2_king_pieces_count: int = np.size(np.where(board == game_state.waiting_player.king_piece)) // 2

        if strong:
            return player_1_normal_pieces_count + 1.5 * player_1_king_pieces_count - \
                   player_2_normal_pieces_count - 1.5 * player_2_king_pieces_count

        return player_1_normal_pieces_count + player_1_king_pieces_count - \
               player_2_normal_pieces_count - player_2_king_pieces_count

    def alpha_beta(self, game_state: GameState, depth: int, alpha: float, beta: float, maximizing_player: bool) \
            -> tuple[list, float]:
        """
        An implementation of the Alpha-Beta algorithm.
        :param game_state: A GameState instance.
        :param depth: The depth of the search.
        :param alpha: The value for alpha.
        :param beta: The value for beta.
        :param maximizing_player: A boolean value, representing whether or not the current call
        tries to maximize the player.
        :return: A tuple of two values - a list of the best next moves and the evaluation.
        """

        if not depth or self.test_winner(game_state=game_state):
            strong: bool

            if (self.game_state.current_player.type == PlayerType.COMPUTER and
                    self.game_state.waiting_player.type == PlayerType.COMPUTER):
                strong = self.game_state.current_player == self._player_1
            else:
                strong = True

            return [], self._evaluate_state(game_state=game_state,
                                            strong=strong)

        game_state_copy: GameState = deepcopy(game_state)

        next_moves, moves_with_capture = game_state.all_moves, game_state.moves_with_capture

        moves_used: dict = moves_with_capture if len(moves_with_capture) else next_moves

        best_moves: list[tuple[Point, Point]] = list()
        best_evaluation: float

        if maximizing_player:
            best_evaluation = float('-inf')

            for point_from in moves_used:
                points_to: dict = moves_used[point_from]

                for point_to in points_to:
                    new_game_state: GameState = deepcopy(game_state_copy)

                    self.make_move(point_from=copy(point_from),
                                   point_to=copy(point_to),
                                   game_state=new_game_state)

                    _, evaluation = self.alpha_beta(game_state=new_game_state,
                                                    depth=depth - 1,
                                                    alpha=alpha,
                                                    beta=beta,
                                                    maximizing_player=False)

                    best_evaluation = max(best_evaluation, evaluation)

                    if best_evaluation > alpha:
                        alpha = best_evaluation
                        best_moves = [(point_from, point_to)]

                    elif best_evaluation == alpha:
                        best_moves.append((point_from, point_to))

                    if beta <= alpha:
                        break

        else:
            best_evaluation = float('+inf')

            for point_from in moves_used:
                points_to: dict = moves_used[point_from]

                for point_to in points_to:

                    new_game_state: GameState = deepcopy(game_state_copy)

                    self.make_move(point_from=copy(point_from),
                                   point_to=copy(point_to),
                                   game_state=new_game_state)

                    _, evaluation = self.alpha_beta(game_state=new_game_state,
                                                    depth=depth - 1,
                                                    alpha=alpha,
                                                    beta=beta,
                                                    maximizing_player=True)

                    best_evaluation = min(best_evaluation, evaluation)

                    if best_evaluation < beta:
                        beta = best_evaluation
                        best_moves = [(point_from, point_to)]

                    elif best_evaluation == beta:
                        best_moves.append((point_from, point_to))

                    if beta <= alpha:
                        break

        self.current_nodes_count += sum(len(m) for m in moves_used.values())

        return best_moves, best_evaluation

    def mini_max(self, game_state: GameState, depth: int, maximizing_player: bool) -> tuple[list, float]:
        """
        An implementation of the Mini-Max algorithm.
        :param game_state: A GameState instance.
        :param depth: The depth of the search.
        :param maximizing_player: A boolean value, representing whether or not the current call
        tries to maximize the player.
        :return: A tuple of two values - a list of the best next moves and the evaluation.
        """

        if not depth or self.test_winner(game_state=game_state):
            strong: bool

            if (self.game_state.current_player.type == PlayerType.COMPUTER and
                    self.game_state.waiting_player.type == PlayerType.COMPUTER):
                strong = self.game_state.current_player == self._player_1
            else:
                strong = True

            return [], self._evaluate_state(game_state=game_state,
                                            strong=strong)

        game_state_copy: GameState = deepcopy(game_state)

        next_moves, moves_with_capture = game_state.all_moves, game_state.moves_with_capture

        moves_used: dict = moves_with_capture if len(moves_with_capture) else next_moves

        best_moves: list[tuple[Point, Point]] = list()
        best_evaluation: float

        if maximizing_player:
            best_evaluation = float('-inf')

            for point_from in moves_used:
                points_to: dict = moves_used[point_from]

                for point_to in points_to:
                    new_game_state: GameState = deepcopy(game_state_copy)

                    self.make_move(point_from=copy(point_from),
                                   point_to=copy(point_to),
                                   game_state=new_game_state)

                    _, evaluation = self.mini_max(game_state=new_game_state,
                                                  depth=depth - 1,
                                                  maximizing_player=False)

                    if evaluation > best_evaluation:
                        best_evaluation = evaluation
                        best_moves = [(point_from, point_to)]

                    elif evaluation == best_evaluation:
                        best_moves.append((point_from, point_to))

        else:
            best_evaluation = float('+inf')

            for point_from in moves_used:
                points_to: dict = moves_used[point_from]

                for point_to in points_to:

                    new_game_state: GameState = deepcopy(game_state_copy)

                    self.make_move(point_from=copy(point_from),
                                   point_to=copy(point_to),
                                   game_state=new_game_state)

                    _, evaluation = self.mini_max(game_state=new_game_state,
                                                  depth=depth - 1,
                                                  maximizing_player=True)

                    if evaluation < best_evaluation:
                        best_evaluation = evaluation
                        best_moves = [(point_from, point_to)]

                    elif evaluation == best_evaluation:
                        best_moves.append((point_from, point_to))

        self.current_nodes_count += sum(len(m) for m in moves_used.values())

        return best_moves, best_evaluation
