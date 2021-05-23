from GUI import GUI
from GameEngine import GameState, GameEngine
from PlayerType import *

from statistics import mean, median

from timeit import default_timer as timer


def handle_input(to_show: str, min_value: int, max_value: int) -> int:
    answer: int

    try:
        answer = int(input(to_show))
    except ValueError:
        print('The value should be an integer.')
        return handle_input(to_show=to_show,
                            min_value=min_value,
                            max_value=max_value)

    while answer < min_value or answer > max_value:
        print('Incorrect answer. Please try again.')
        answer = int(input('> '))

    return answer


def handle_statistics(arr: list, name: str) -> None:

    if not arr:
        return

    arr = sorted(arr)

    print(f'{name}: ')

    print(f'Min: {arr[0]}')
    print(f'Max: {arr[-1]}')

    print(f'Mean: {mean(arr)}')
    print(f'Median: {median(arr)}')

    print('\n\n')


if __name__ == '__main__':

    player_1_type: PlayerType
    player_2_type: PlayerType

    player_1_name: str
    player_2_name: str

    use_alpha_beta: bool = True

    search_alg_depth: int = 0
    flip: bool = False

    with_animations: bool = True

    print('Welcome!\n\n')

    game_type = handle_input('Choose the game type:'
                             '\n1. Human vs Human'
                             '\n2. Human vs Computer'
                             '\n3. Computer vs Computer\n> ', 1, 3)

    if game_type == 3:
        player_1_type = player_2_type = PlayerType.COMPUTER
        player_1_name = 'Computer 1'
        player_2_name = 'Computer 2'
        search_alg_depth = 4

    else:
        player_1_type = PlayerType.HUMAN

        if game_type == 1:
            player_1_name, player_2_name = input('Player 1\'s name: '), input('Player 2\'s name: ')
            player_2_type = PlayerType.HUMAN
        else:
            player_1_name = input('Your name: ')
            player_2_name = 'Computer'
            player_2_type = PlayerType.COMPUTER

            search_alg: int = handle_input('What search algorithm to use for the computer:'
                                           '\n1. Alpha-Beta'
                                           '\n2. Mini-Max\n> ', 1, 2)

            if search_alg == 2:
                use_alpha_beta = False

            difficulty: int = handle_input('Difficulty:'
                                           '\n1. Easy'
                                           '\n2. Medium'
                                           '\n3. Hard\n> ', 1, 3)

            if difficulty == 1:
                search_alg_depth = 2
            elif difficulty == 2:
                search_alg_depth = 4
            else:
                search_alg_depth = 6

        start_first: int = handle_input('Who should start first:'
                                        f'\n1. {player_1_name}'
                                        f'\n2. {player_2_name}\n> ', 1, 2)

        if start_first == 2:
            player_1_name, player_2_name = player_2_name, player_1_name
            player_1_type, player_2_type = player_2_type, player_1_type

            if game_type == 2:
                flip = True

        animations: int = handle_input('Use animations:'
                                       '\n1. Yes'
                                       '\n2. No\n> ', 1, 2)

        with_animations = animations == 1

    game_state: GameState = GameState()
    game_engine: GameEngine = GameEngine(game_state=game_state,
                                         use_alpha_beta=use_alpha_beta,
                                         player_1_name=player_1_name,
                                         player_2_name=player_2_name,
                                         player_1_type=player_1_type,
                                         player_2_type=player_2_type,
                                         search_algorithm_depth=search_alg_depth,
                                         flip=flip)

    gui: GUI = GUI(screen_size=800,
                   max_fps=60,
                   game_engine=game_engine,
                   with_animations=with_animations)

    start_time: float = timer()

    gui.run()

    end_time: float = timer()

    print()

    print(f'The game has been played for {end_time - start_time} seconds.\n')

    print(f'{player_1_name} moves: {gui.first_player_moves}')
    print(f'{player_2_name} moves: {gui.second_player_moves}')

    print()

    handle_statistics(gui.game_engine.search_algorithm_number_of_nodes_each_turn, 'Number of nodes')
    handle_statistics(gui.first_player_times, f'{player_1_name}\'s times')
    handle_statistics(gui.second_player_times, f'{player_2_name}\'s times')
