#!/usr/bin/env python
"""Returns squared difference between a parameter value and 5."""
import argparse

parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('value_file', type=str)
parser.add_argument('--params_file', type=str)
flags = parser.parse_args()


def main():  # noqa

    lines = open(flags.params_file).read().split('\n')
    val = None
    for line in lines:
        if 'param1\t' in line:
            val = line.split('\t')[1]
    if val is None:
        raise ValueError('Param value not found.')
    evaluation = str(-1 * (float(val) - 5) ** 2)
    with open(flags.value_file, 'wb') as w:
        w.write(evaluation.encode())


if __name__ == '__main__':
    main()
