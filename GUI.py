import pygame as pg
from GameEngine import Piece, PlayerType, GameEngine
from Point import Point

from timeit import default_timer as timer

from tkinter import *
from tkinter import messagebox


def _load_image(name: str):
    """
    Loads an image using pygame's "load" function.
    :param name: path to the image.
    :return: the image.
    """
    return pg.image.load(f'assets/images/{name}.png')


def _get_notation(point: Point) -> str:
    """
    Gets the notation of a piece.
    :param point: Piece's coordinates as a Point object.
    :return: A string, representing the notation.
    """

    return str(4 * point.row + (point.column >> 1) + 1)


def print_move(point_from: Point, point_to: Point):
    """
    Prints the move in Checkers notation.

    :param point_from: Source point.
    :param point_to: Destination point.
    :return: None.
    """

    print(f'{_get_notation(point=point_from)} -> {_get_notation(point=point_to)}.')


class GUI:
    width: int
    height: int
    board_size: int

    max_fps: int

    square_size: int
    square_colors: list

    selected_square: Point

    images: dict

    screen: pg.display
    clock: pg.time.Clock

    running: bool

    game_engine: GameEngine

    change_in_coordinates: list[Point]

    correct_next_moves: list[Point]

    with_animations: bool

    timer: float

    first_player_name: str
    second_player_name: str

    first_player_times: list
    second_player_times: list

    first_player_moves: int
    second_player_moves: int

    currently_thinking: bool

    def __init__(self, screen_size: int, max_fps: int, game_engine: GameEngine, with_animations: bool = True) -> None:
        """
        Initializes the object.

        :param screen_size: The width and the height of the screen.
        :param max_fps: The maximum number of frames per second.
        :param game_engine: An instance of the class GameEngine.
        :param with_animations: A boolean value, indicating whether or not to animate the moves.
        """

        pg.display.set_caption('Rusu Andrei-Cristian - Checkers')

        self.width = self.height = screen_size
        self.max_fps = max_fps
        self.game_engine = game_engine

        self.with_animations = with_animations

        self.board_size = 8
        self.square_size = self.height // self.board_size
        self.images = dict()

        self.square_colors = [pg.Color('white'), pg.Color('gray')]

        self.screen = pg.display.set_mode((self.width, self.height))

        self.clock = pg.time.Clock()

        self.screen.fill(pg.Color("black"))

        self.selected_square = Point()

        self.running = False

        self.correct_next_moves = list()

        self.human_after_computer = False

        self.first_player_name = self.game_engine.game_state.current_player.name
        self.second_player_name = self.game_engine.game_state.waiting_player.name

        self.first_player_times = list()
        self.second_player_times = list()

        self.first_player_moves = 0
        self.second_player_moves = 0

        self.currently_thinking = False

    def _scale_image(self, image: pg.image) -> pg.transform:
        """
        Scales an image to the size of a "square" in the screen.
        :param image: A pg.image object.
        :return: The scaled image.
        """

        return pg.transform.scale(image, (self.square_size, self.square_size))

    #

    def _rect_constructor(self, row: float, column: float) -> pg.Rect:
        """
        Constructs a rectangle.
        :param row: The row at which to start.
        :param column: The column at which to end.
        :return: A pg.Rect instance, of size (self.square_size, self.square_size), starting at (row, column).
        """

        return pg.Rect(column * self.square_size, row * self.square_size, self.square_size, self.square_size)

    def _draw_rect(self, color: pg.Color, row: float = None, column: float = None, rect: pg.Rect = None) -> None:
        """
        Draws a rectangle.
        :param color: The color of the rectangle.
        :param row: The row at which to start the rectangle.
        :param column: The column at which to start the rectangle.
        :param rect: An instance of pg.Rect or None (in which case, it is created with _rect_constructor())
        :return: None
        """

        if rect is None:
            rect = self._rect_constructor(row, column)

        pg.draw.rect(self.screen, color, rect)

    def _draw_piece(self, piece: Piece, row: int, column: int) -> None:
        """
        Draws a piece.
        :param piece: Instance of Piece.
        :param row: The row at which to start the piece.
        :param column: The column at which to start the piece.
        :return: None
        """

        self.screen.blit(self.images[piece], self._rect_constructor(row, column))

    #

    def _reset_correct_next_moves(self):
        """
        Resets the highlight for the next correct moves.
        :return: None
        """

        for possible_square in self.correct_next_moves:
            self._draw_rect(self.square_colors[(possible_square.row + possible_square.column) & 1],
                            possible_square.row, possible_square.column)

        self.correct_next_moves.clear()
        self.draw_board()
        self.draw_pieces()

    def _show_correct_next_moves(self, row: int, column: int) -> None:
        """
        Highlight the next correct moves.
        :param row: The row for the clicked piece.
        :param column: The column for the clicked piece.
        :return: None
        """

        self._reset_correct_next_moves()

        surface: pg.Surface = pg.Surface((self.square_size, self.square_size))

        surface.set_alpha(64)
        surface.fill(pg.Color('blue'))

        self.screen.blit(surface, (column * self.square_size, row * self.square_size))
        self._draw_piece(self.game_engine.game_state.board[row][column], row, column)

        surface.set_alpha(100)
        surface.fill(pg.Color('yellow'))

        for point in self.game_engine.get_coordinates_valid_moves(row, column):
            new_row: int = point.row
            new_column: int = point.column

            self.screen.blit(surface, (new_column * self.square_size, new_row * self.square_size))

            self.correct_next_moves.append(Point(new_row, new_column))

    def _animate_move(self, point_from: Point, point_to: Point, piece_from: Piece, jumped_piece: Piece,
                      is_undo: bool = False) -> None:

        """
        Animates a move.
        :param point_from: A Point instance, representing the point from which to move.
        :param point_to: A Point instance, representing the point at which to go.
        :param piece_from: A Piece instance, representing the piece that was located at point_from.
        :param jumped_piece: A Piece instance.
        :param is_undo: A boolean value, representing whether or not the animation was called by undo().
        :return: None
        """

        d_row: int = point_to.row - point_from.row
        d_column: int = point_to.column - point_from.column

        frames_per_square: int = 5
        total_frame_count: int = (abs(d_row) + abs(d_column)) * frames_per_square

        from_row: int = point_from.row
        from_column: int = point_from.column

        to_row: int = point_to.row
        to_column: int = point_to.column

        color_to_replace: pg.Color = self.square_colors[(from_row + from_column) & 1]

        row_jumped: int = (from_row + to_row) >> 1
        column_jumped: int = (from_column + to_column) >> 1

        from_piece_image: pg.image = self.images[piece_from]

        matched: bool = False

        for frame in range(total_frame_count + 1):
            row: float
            column: float

            self.draw_board()
            self.draw_pieces()

            row, column = from_row + d_row * frame / total_frame_count, \
                          from_column + d_column * frame / total_frame_count

            if not matched and row == row_jumped and column == column_jumped:
                matched = True

            if jumped_piece is not None and jumped_piece != Piece.EMPTY:
                if not matched and not is_undo or matched and is_undo:
                    self._draw_piece(jumped_piece, row_jumped, column_jumped)
                else:
                    self._draw_rect(color_to_replace, row_jumped, column_jumped)

            self._draw_rect(color_to_replace, point_to.row, point_to.column)

            self.screen.blit(from_piece_image, self._rect_constructor(row, column))

            self.clock.tick(60)
            pg.display.flip()

    #

    def _handle_click(self, row: int, column: int):
        """
        Click handler. Highlights the next correct moves or executes a move.
        :param row: The row of the clicked piece.
        :param column: The column of the clicked piece.
        :return: None
        """

        if not self.selected_square.is_null() and Point(row, column) in \
                self.game_engine.get_coordinates_valid_moves(self.selected_square.row, self.selected_square.column):

            point_to: Point = Point(row, column)

            piece_from, jumped_piece = self.game_engine.make_move(point_from=self.selected_square,
                                                                  point_to=point_to)

            print_move(point_from=self.selected_square,
                       point_to=point_to)

            end_time: float = timer()

            print('Thinking time: %.5f.\n' % (end_time - self.timer))

            thinking_time: float = end_time - self.timer

            if self.game_engine.game_state.current_player.name == self.first_player_name:
                self.second_player_times.append(thinking_time)
                self.second_player_moves += 1
            else:
                self.first_player_times.append(thinking_time)
                self.first_player_moves += 1

            self.currently_thinking = False

            if self.with_animations:
                self._animate_move(point_from=self.selected_square,
                                   point_to=Point(row, column),
                                   piece_from=piece_from,
                                   jumped_piece=jumped_piece)

            self._reset_correct_next_moves()
            self.draw_board()

            self.timer = timer()

            return self.draw_pieces()

        if self.game_engine.game_state.board[row][column] not in \
                (self.game_engine.game_state.current_player.normal_piece,
                 self.game_engine.game_state.current_player.king_piece):
            return

        if not self.selected_square.equals(row, column):
            self.selected_square.set_coords(row, column)
            return self._show_correct_next_moves(row, column)

        self.selected_square.reset()
        self._reset_correct_next_moves()

    def _handle_undo(self):
        """
        Undoes the last move.
        :return: None
        """

        undo = self.game_engine.undo()

        if undo is None:
            return

        point_from, point_to, piece_from, jumped_piece = undo

        if self.with_animations:
            self._animate_move(point_to, point_from, piece_from, jumped_piece, is_undo=True)

        if piece_from in (Piece.RED_NORMAL, Piece.RED_KING):
            self.first_player_moves -= 1
        else:
            self.second_player_moves -= 1

        self.selected_square = Point()
        self.draw_board()
        self.draw_pieces()

        if self.game_engine.game_state.current_player.type == PlayerType.COMPUTER:
            self._handle_undo()

    def _handle_redo(self):
        """
        Redoes the last action.
        :return: None
        """

        redo = self.game_engine.redo()

        if redo is None:
            return

        point_from, point_to, piece_from, jumped_piece = redo

        if self.with_animations:
            self._animate_move(point_from, point_to, piece_from, jumped_piece)

        if piece_from in (Piece.RED_NORMAL, Piece.RED_KING):
            self.first_player_moves += 1
        else:
            self.second_player_moves += 1

        self.selected_square = Point()
        self.draw_board()
        self.draw_pieces()

        if self.game_engine.game_state.current_player.type == PlayerType.COMPUTER:
            self._handle_redo()

    def _handle_reset(self):
        """
        Resets the game.
        :return: None
        """

        Tk().wm_withdraw()

        reset: str = messagebox.askquestion(title='Reset',
                                            message='Are you sure you want to reset the game?')

        if reset == 'no':
            return

        self.game_engine.reset()
        self.draw_board()
        self.draw_pieces()
        self.selected_square = Point()
        self.first_player_times = list()
        self.second_player_times = list()
        self.first_player_moves = 0
        self.second_player_moves = 0

    def _handle_winner(self):
        """
        Shows the name of the winner, if the game has ended.
        :return: True if there is a winner and False otherwise.
        """

        if self.game_engine.test_winner():
            print(f'\n\n{self.game_engine.game_state.waiting_player.name} won!')
            self.running = False
            return True

        return False

    def _handle_events(self):
        """
        The main event handler for interactions with the GUI.
        :return: None.
        """

        if not self.currently_thinking:
            print(f'{self.game_engine.game_state.current_player.name}\'s turn:')
            self.currently_thinking = True
            self.timer = timer()

        for event in pg.event.get():

            if self.game_engine.game_state.current_player.type == PlayerType.COMPUTER:

                point_from, point_to, piece_from, piece_jumped = self.game_engine.make_best_move()

                print_move(point_from=point_from,
                           point_to=point_to)

                end_time: float = timer()

                thinking_time: float = end_time - self.timer

                print('Thinking time: %.5f.\n' % thinking_time)

                if self.game_engine.game_state.current_player.name == self.first_player_name:
                    self.second_player_times.append(thinking_time)
                    self.second_player_moves += 1
                else:
                    self.first_player_times.append(thinking_time)
                    self.first_player_moves += 1

                self.currently_thinking = False

                if self.with_animations:
                    self._animate_move(point_from, point_to, piece_from, piece_jumped)

                if self._handle_winner():
                    break

                self.draw_board()
                self.draw_pieces()

            if event.type == pg.QUIT:
                self.running = False

            elif event.type == pg.MOUSEBUTTONDOWN:
                location: pg.mouse = pg.mouse.get_pos()

                column: int = location[0] // self.square_size
                row: int = location[1] // self.square_size

                self._handle_click(row=row,
                                   column=column)

                if self._handle_winner():
                    break

            elif event.type == pg.KEYDOWN:
                if event.key == pg.K_r:
                    self._handle_reset()

                elif pg.key.get_mods() & pg.KMOD_CTRL and event.key == pg.K_z:
                    self._handle_undo()

                elif pg.key.get_mods() & pg.KMOD_CTRL and event.key == pg.K_y:
                    self._handle_redo()

        self.clock.tick(self.max_fps)
        pg.display.flip()

    #

    def load_images(self):
        """
        Loads all the images and saves them into self.images.
        :return: None
        """

        self.images[Piece.RED_NORMAL] = self._scale_image(_load_image("red_piece"))
        self.images[Piece.WHITE_NORMAL] = self._scale_image(_load_image("white_piece"))
        self.images[Piece.WHITE_KING] = self._scale_image(_load_image("white_piece_king"))
        self.images[Piece.RED_KING] = self._scale_image(_load_image("red_piece_king"))

    def draw_board(self):
        """
        Draws the board, without the pieces.
        :return: None.
        """

        for row in range(self.board_size):
            for column in range(self.board_size):
                self._draw_rect(self.square_colors[(row + column) & 1], row, column)

    def draw_pieces(self):
        """
        Draws the pieces on the board.
        :return: None.
        """

        for row in range(self.board_size):
            for column in range(self.board_size):
                current_piece = self.game_engine.game_state.board[row][column]

                if current_piece != Piece.EMPTY:
                    self._draw_piece(current_piece, row, column)

    #

    def run(self):
        """
        The main function. Loads the assets, draws the board and the pieces
        and then loops while the game hasn't ended or the windows wasn't closed.
        :return: None.
        """

        pg.init()

        self.load_images()
        self.draw_board()
        self.draw_pieces()

        self.running = True

        while self.running:
            self._handle_events()
