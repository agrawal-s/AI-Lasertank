#!/usr/bin/python

import sys
from laser_tank import LaserTankMap

"""
Tester script.

Use this script to test whether your output files are valid solutions. You
should avoid modifying this file directly.

You may import methods from this file into your solution if you wish.

COMP3702 2020 Assignment 1 Support Code

Last updated by njc 16/08/20
"""


def get_optimal_number_of_steps(filename):
    """
    Get the number of steps for an optimal solution for the given testcase file.
    :param filename: name of testcase file
    :return: number of steps in optimal solution
    """
    f = open(filename, 'r')
    steps = int(f.readline().strip())
    f.close()
    return steps


def get_time_limit(filename):
    """
    Get the time limit for the given testcase file (in seconds).
    :param filename: name of testcase file
    :return: amount of time given to solve this
    """
    f = open(filename, 'r')
    _ = f.readline()
    time_limit = float(f.readline().strip())
    f.close()
    return time_limit


def main(arglist):
    """
    Test whether the given output file is a valid solution to the given map file.
    :param arglist: map file name, output file name
    """
    if len(arglist) != 2:
        print("Running this file tests whether the given output file is a valid solution to the given map file.")
        print("Usage: tester.py [map_file_name] [output_file_name]")
        return

    map_file = arglist[0]
    soln_file = arglist[1]

    optimal_steps = get_optimal_number_of_steps(map_file)
    game_map = LaserTankMap.process_input_file(map_file)

    f = open(soln_file, 'r')
    moves = f.readline().strip().split(',')

    # apply each move in sequence
    error_occurred = False
    for i in range(len(moves)):
        move = moves[i]
        ret = game_map.apply_move(move)
        if ret == LaserTankMap.COLLISION:
            print("ERROR: Move resulting in Collision performed at step " + str(i))
            error_occurred = True
        elif ret == LaserTankMap.GAME_OVER:
            print("ERROR: Move resulting in Game Over performed at step " + str(i))
            error_occurred = True

    if error_occurred:
        return -1

    if game_map.is_finished():
        print("Puzzle solved.")
        if len(moves) == optimal_steps:
            print("Solution is optimal (" + str(len(moves)) + " steps)!")
            return 0
        else:
            print("Solution is " + str(len(moves) - optimal_steps) + " steps longer than optimal.")
            return len(moves) - optimal_steps
    else:
        print("ERROR: Goal not reached after all actions performed.")
        return -1


if __name__ == '__main__':
    ret = main(sys.argv[1:])
    sys.exit(ret)

