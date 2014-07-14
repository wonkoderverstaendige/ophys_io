#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Jun 15, 2014 23:40
@author: <'Ronny Eichler'> ronny.eichler@gmail.com

Reads Open Ephys .continuous files and converts them to raw 16-bit .dat files readable by
"old" analysis tools (KlustaKwik, Spike_Detekt, Klusters, Neuroscope)
"""
from __future__ import division
import os
import re
import numpy as np
from collections import Iterable


SIZE_HEADER = 1024  # size of header in B
NUM_SAMPLES = 1024  # number of samples per record
SIZE_RECORD = 2070  # total size of record (2x1024 B samples + record header)
REC_MARKER = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 255], dtype=np.uint8)


class SessionReader(object):
    def __init__(self, target, channels=64, proc_node=100):
        self.channels = channels if channels is Iterable else range(channels)
        self.files = {set_id: files for set_id, files in enumerate(target)}


def gather_files(input_directories, channels, proc_node):
    """Return list of paths to valid input files from list of input directories."""
    #TODO: Handling non-existing items in file list (i.e. directories named like .continuous files)

    base_names = [os.path.join(path, '{proc_node:d}_CH{channel:d}.continuous') for path in input_directories]
    filenames = [fname_base.format(proc_node=proc_node, channel=chan) for chan in channels for fname_base in base_names]

    for f in filenames:
        print 'Is {file} an existing file? {existance}'.format(file=f, existance='YES' if os.path.isfile(f) else 'NO!')
        assert os.path.isfile(f)

    return filenames


def read_header(filename, size_header=SIZE_HEADER):
    """Return dict with .continuous file header content."""
    # TODO: Compare headers, should be identical except for channel

    # 1 kiB header string data type
    header_dt = np.dtype([('Header', 'S%d' % size_header)])
    header = read_segment(filename, offset=0, count=1, dtype=header_dt)

    # Stand back! I know regex!
    # Annoyingly, there is a newline character missing in the header (version/header_bytes)
    regex = "header\.([\d\w\.\s]{1,}).=.\'*([^\;\']{1,})\'*"
    header_str = str(header[0][0]).rstrip(' ')
    header_dict = {group[0]: group[1] for group in re.compile(regex).findall(header_str)}
    for key in ['bitVolts', 'sampleRate']:
        header_dict[key] = float(header_dict[key])
    for key in ['blockLength', 'bufferSize', 'header_bytes', 'channel']:
        header_dict[key] = int(header_dict[key] if not key == 'channel' else header_dict[key][2:])

    return header_dict


def read_records(filename, record_offset=0, record_count=10000,
                 size_header=SIZE_HEADER, num_samples=NUM_SAMPLES, size_record=SIZE_RECORD):

    # data type of individual records, n Bytes              # (2048 + 22) Byte = 2070 Byte total
    data_dt = np.dtype([('timestamp', np.int64),            # 8 Byte
                        ('n_samples', np.uint16),           # 2 Byte
                        ('rec_num', np.uint16),             # 2 Byte
                        ('samples', ('>i2', num_samples)),  # 2 Byte each x 1024; note endian type
                        ('rec_mark', (np.uint8, 10))])      # 10 Byte

    return read_segment(filename, offset=size_header + record_offset*size_record, count=record_count, dtype=data_dt)


def read_segment(filename, offset, count, dtype):
    """Read segment of a file from [offset] for [count]x[dtype]"""
    # TODO: Use alternative reading method when filename is a file id instead of filename
        # alternative which moves file pointer
        #fid = open(filename, 'rb')
        #data = fid.read(STUFF)
        #fid.close()

    with open(filename, 'rb') as fid:
        fid.seek(offset)
        segment = np.fromfile(fid, dtype=dtype, count=count)

    return segment


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


def data_to_buffer(file_path, channels, count=1000, proc_node=100, buf=None, size_record=SIZE_RECORD):
    """Read [count] records from [proc_node] file at [filepath] into a buffer."""
    assert channels is Iterable
    
    # temporary storage
    buf = buf if buf is not None else np.zeros((len(channels), count * size_record), dtype='>i2')

    # load chunk of data from all channels
    # for n in channels:
    #     filename = file_path+base_end.format(proc_node=proc_node, channel=n+1)
    #     # channel offset is clunky shortcut for plotting
    #     if channel_offset:
    #         buf[n] = read_records(filename, count=count)['samples'].ravel() + channel_offset * (n-32)
    #     else:
    #         buf[n] = read_records(filename, count=count)['samples'].ravel()

    return buf


def folders_to_dat(in_dir, outfile, channels, proc_node=100, *args, **kwargs):
    """Given list of input directories [in_dir] will write out .continuous files for channels in [channels] iterable
    of an open ephys [proc_node] into single [outfile] .dat file.
    """
    #TODO: out_log file should be ast.eval()-uable (i.e. exported as dict)

    files = gather_files(in_dir, channels, proc_node)
    log_string = '{in_dir}:\nChannels: {channels}, proc_node: {proc_node}\n\n'
    try:
        with open(outfile, 'wb') as out_fid_dat, open(outfile+'.log', 'w') as out_fid_log:
            out_fid_log.write(log_string.format(in_dir=in_dir, channels=str(channels), proc_node=proc_node))

            for f in files:
                out_fid_log.write(f+':\n'+str(read_header(f))+'\n\n')
                #out_fid_dat.write()

    except IOError as e:
        print 'Operation failed: %s' % e.strerror

    return files


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
                        default='raw.dat',
                        help="Output file path.")
    args = parser.parse_args()
    print args
    channels = args.channel_list if args.channel_list else list(range(1, args.channel_count+1))

    print folders_to_dat(in_dir=args.target,
                            outfile=args.output,
                            channels=channels)
