from board import Board
import argparse


def main(args):
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--size',
        type=int,
        help='Size of the board.'
    )
    main(parser.parse_args())
    