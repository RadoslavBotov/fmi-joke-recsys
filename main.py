import os
import argparse
from typing import Dict

from src.net import net_calcs
from src.svd import svd_calcs
from src.tfidf import tfidf_calcs


def init_parser() -> argparse.ArgumentParser:
    """
    Initializes the command-line argument parser.
    """
    parser = argparse.ArgumentParser(
        prog='ProgramName',
        description='Selects and executes a recommendation algorithm.',
        epilog='Example: python main.py -a nn -d 1 -n train'
    )

    parser.add_argument('-a', '--algo', type=str, required=True, choices=['tfidf', 'svd', 'nn'], help='Algorithm to execute')
    parser.add_argument('-d', '--data', type=int, required=True, choices=[1, 2, 3], help='Dataset choice (1: jester_data_1_sdv.csv, 2: jester_data_2_sdv.csv, or 3: jester_data_comb_sdv.csv)')
    parser.add_argument('-n', '--nn_mode', type=str, required=False, choices=['train', 'test'], default='train', help='Train or Test a Neural Net')
    parser.add_argument('-t', '--tfidf_mode', type=str, required=False, choices=['tf', 'bow'], default='tf', help='Use TF-IDF or BOW')

    return parser

def get_parser_args(parser: argparse.ArgumentParser) -> Dict[str, str|int]:
    """
    Parses command-line arguments and converts them into a dictionary.
    """
    args = parser.parse_args()
    return vars(args)

def parser_data_to_path(args_dict: Dict[str, str|int]) -> None:
    """
    Maps the user-provided integer data index to an actual file path.
    Modifies args_dict in-place to store the string path.
    """
    data_map = {
        1: os.path.join('data_svd', 'jester_data_1_svd.csv'),
        2: os.path.join('data_svd', 'jester_data_2_svd.csv'),
        3: os.path.join('data_svd', 'jester_data_comb_svd.csv')
    }
    args_dict['data'] = data_map.get(args_dict['data'])

def execute_algo(args_dict: Dict[str, str|int]) -> None:
    """
    Routes the execution to the appropriate algorithm module based on the user's choice.
    """
    match args_dict['algo']:
        case 'tfidf':
            tfidf_calcs(args_dict)
        case 'svd':
            svd_calcs(args_dict)
        case 'nn':
            net_calcs(args_dict)

def run_main() -> None:
    # Initialize and parse arguments
    parser = init_parser()
    args_dict = get_parser_args(parser)

    # Transform raw inputs into functional file paths
    parser_data_to_path(args_dict)
    print(f"Executing with configuration: {args_dict}")

    # Run the selected algorithm
    execute_algo(args_dict)

if __name__ == "__main__":
    run_main()

# python .\main.py -a tfidf|svd|nn -d 1|2|3 -n train|test -t tf|bow
# =================================================================
# python .\main.py -a tfidf -d 1|2|3 -t tf
# python .\main.py -a tfidf -d 1|2|3 -t bow

# python .\main.py -a svd -d 1|2|3

# python .\main.py -a nn -d 1|2|3 -n train
# python .\main.py -a nn -d 1|2|3 -n test
