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
    if len(sys.argv) != 2:
        print("ERROR : The program need at least one argument FILE_NAME to "
            "execute itself.")
        sys.exit(1)

    output_file = sys.argv[1]
    # no -S because this is just a test we don't want to save the result
    cmd = "main.py networks/saved/testnw3 -ls 150 -ts 100 -gdf Constant0.5"

    print("Compiling the command :", cmd, "\n")

    os.system("python3 -m cProfile -o profile/dataProfile " + cmd)

    profile = "profile/dataProfile"
    # in case the output file doesn't exist, we have to create it
    if not os.path.isfile(output_file):
        os.system("touch " + output_file)

    # all the information will be printed there thanks to print_stats()
    sys.stdout = open(output_file, "w")

    # creation of the information table
    p = pstats.Stats(profile)
    p.strip_dirs().sort_stats("cumtime").print_stats()

    # reset stdout to normal
    sys.stdout = sys.__stdout__


if __name__ == '__main__':
    main()
