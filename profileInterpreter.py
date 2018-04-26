#!/usr/bin/env python3

"""
    Intepreter to use cProfile and analyse the results.
    Use to optimize code by analazing the execution time of each functions.
"""

import sys, os, pstats
import StringIO

def main():
    """
        Main function.
    """
    profile = sys.argv[1]
    s = StringIO.StringIO()
    p = pstats.Stats(profile, stream=s)
    p.strip_dirs().sort_stats("cumtime").print_stats()
    ps.dump_stats("stats.dmp")


if __name__ == '__main__':
    main()
