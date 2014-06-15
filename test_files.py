#!/usr/bin/env python

# Generate test directories to mess with from a list of filenames.

import argparse
import os
import sys

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input')
parser.add_argument('-t', '--target')

args = parser.parse_args()

base_dir = args.target if args.target else 'testing'

input_file = args.input if args.input is not None else 'filenames.tsv'
with open(input_file) as fh:
    [os.makedirs(os.path.join(base_dir, row.split()[0])) for row in fh.readlines()]
            
