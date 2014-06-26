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
    fnames = []
    for fname_base in base_names:
        print channels
        fnames.extend([fname_base.format(proc_node=proc_node, channel=chan) for chan in channels])
    return fnames 

def read_header(filename):
    """Return dict with raw files header content."""
    # TODO: Use alternative reading method when filename is a file id instead of filename

    # 1 kiB header string
    header_dt = np.dtype([('Header', 'S%d', % SIZE_HEADER)])
    header = np.fromfile(filename, dtype=header_dt, count=1)

    # alternative which moves file pointer
    #fid = open(filename, 'rb')
    #header = fid.read(SIZE_HEADER)
    #fid.close()

    # Stand back! I know regex!
    # Annoyingly, there is a newline character missing in the header
    regex = "header\.([\d\w\.\s]{1,}).=.\'*([^\;\']{1,})\'*"
    header_str = str(header[0][0]).rstrip(' ')
    header_dict = {group(0): group[1] for group in re.compile(regex).findall(header_str)}
    for key in ['bitVolts', 'sampleRate']:
        header_dict[key] = float(header_dict[key])
    for key in ['blockLength', 'bufferSize', 'header_bytes', 'channel']:
        header_dict[key] = int(header_dict[key] if not key == 'channel' else header_dict[key][2:])
    return header_dict

def read_records(filename, offset=0, count=10000):
    with open(filename, 'rb') as fid:
        # move pointer to new position in file
        fid.seek(SIZE_HEADER + offset*SIZE_RECORD)

        # data type of individual records, n Bytes
        data_dt = np.dtype([('timestamp', np.int64),
                            ('n_samples', np.uint16),
                            ('rec_num', np.uint16),
                            # note endian type
                            ('samples', ('>i2', NUM_SAMPLES)),
                            ('rec_mark', (np.uint8, 10))])

       return np.fromfile(fid, dtype=data_dt, count=count)

def check_data(data):
    """Sanity checks of records."""

    # Timestamps should increase monotonically in steps of 1024
    assert len(set(np.diff(data['timestamp']))) == 1 and np.diff(data['timestamp'][:2]) == 1024
    print 'timestamps: ', data['timestamp'][0]

    # Number of samples in each record should be NUM_SAMPLES, or 1024
    assert len(set(data['n_samples'])) == 1 and data['n_samples'][0] == NUM_SAMPLES
    print 'N samples: ', data['n_samples'][0]

    # should be byte pattern [0...8, 255]
    markers = set(map(str, data['rec_mark']))  # <- slow
    assert len(markers) == 1 and str(REC_MARKER) in markers
    print 'record marker: ', data['rec_mark'][0]

    # should be zero, or there are multiple recordings in this file
    assert len(set(data['rec_num'])) == 1 and data['rec_num'][0] == 0
    print 'Number recording: ', data['rec_num'][0]


def data_to_buffer(file_path, channels, count=1000, proc_node=100, buf=None):
    """Read [count] records from [proc_node] file at [filepath] into a buffer."""
    assert channels is Iterable
    
    # temporary storage
    buf = np.zeros((len(channels), count*SIZE_NUM_SAMPLES

def continuous_to_dat(in_dir, outfile, channels, proc_node=100, chunk_size=10000):
    files = gather_files(in_dir, channels, proc_node)
    for f in files:
        print 'Is {file} an existing file? {existance}'.format(file=f, existance='YES' if os.path.isfile(f) else 'NO!')
        assert os.path.isfile(f)

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

    group = parser.add_mutually_exclusive_group()

    group.add_argument("-c", "--channel_list",
                        nargs='*', default=[],
                        type=int,
                        help="""List of channels to merge from the .probe file""")

    group.add_argument("-C", "--channel-count", default=64, type=int,
                       help="Number of consecutive channels, 1-based index. E.g. 64: Channels 1:64")

    parser.add_argument("-l", "--layout", help="Path to .probe file.")
    parser.add_argument("-p", "--params", help="Path to .params file.")

    parser.add_argument("-o", "--output",
                        default='./proc/raw.dat',
                        help="Output file path.")
    args = parser.parse_args()
    print args
    channels = args.channel_list if args.channel_list else list(range(1, args.channel_count+1))

    print continuous_to_dat(in_dir=args.target,
                            outfile=args.output,
                            channels=channels)
