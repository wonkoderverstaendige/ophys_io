#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Jun 15, 2014 23:40
@author: <'Ronny Eichler'> ronny.eichler@gmail.com

Reads Open Ephys .continuous files and converts them to raw 16-bit .dat files readable by
"old" analysis tools (KlustaKwik, Spike_Detekt, Klusters, Neuroscope)
"""
import os
import numpy as np
from collections import Iterable


class SessionReader(object):
    def __init__(self, target, channels=64, proc_node=100):
        self.channels = channels if channels is Iterable else range(channels)
        self.files = {set_id: files for set_id, files in enumerate(target)}

def gather_files(in_dirs, channels, proc_node):
    print in_dirs
    base_names = [os.path.join(path, '{proc_node:d}_CH{channel:d}.continuous') for path in in_dirs]
    return base_names


def continuous_to_dat(in_dir, outfile, channels, proc_node=100, chunk_size=10000):
    files = gather_files(in_dir, channels, proc_node
    fnames = [fname_base.format(proc_node=proc_node, channel=chan+1) for chan in channels]
    for file in files:
        print 'Is {file} an existing file?'.format(file=file)
        assert os.path.isfile(files)

    return files
    with open(outfile, 'wb'):
        pass

if __name__ == "__main__":
    # Command line interface
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("target",
                        nargs='*',
                        help="""Path/list of paths to directories containing raw .continuous data OR path to .session
                                definition file.
                                Listing multiple files will result in data sets being merged in listed order.""")

    parser.add_argument("-c", "--channels",
                        nargs='*', default=64,
                        help="""Number of channels, or comma-separated list of channels to merge from the
                                .probe file""")

    parser.add_argument("-l", "--layout", help="Path to .probe file.")
    parser.add_argument("-p", "--params", help="Path to .params file.")

    parser.add_argument("-o", "--output",
                        default='./proc/raw.dat',
                        help="Output file path.")
    args = parser.parse_args()
    print args
    print continuous_to_dat(args.target, args.output, args.output)