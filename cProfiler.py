#!/usr/bin/env python3

"""
    Intepreter to use cProfile and analyse the results.
    Use to optimize code by analazing the execution time of each functions.
"""

import sys, os, pstats

def main():
    """
        Main function.
    """
    # no -S because this is just a test we don't want to save the result
    cmd = "main.py networks/saved/testnw2 -ls 100 -ts 100 -gdf Constant0.5"

    print("Compiling the command :", cmd, "\n")

    os.system("python3 -m cProfile -o profile/dataProfile " + cmd)
    profile = "profile/dataProfile"
    output_file = "profile/cProfiler2.txt"

    # all the information will be printed there thanks to print_stats()
    sys.stdout = open(output_file, "w")

    # creation of the information table
    p = pstats.Stats(profile)
    p.strip_dirs().sort_stats("cumtime").print_stats()

    # reset stdout to normal
    sys.stdout = sys.__stdout__
    print("\nThe file", output_file, "contains all the information about"
        " the execution time of each function and more.")


if __name__ == '__main__':
    main()
