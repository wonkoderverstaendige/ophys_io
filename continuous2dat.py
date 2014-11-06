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
import time
from collections import Iterable


SIZE_HEADER = 1024  # size of header in B
NUM_SAMPLES = 1024  # number of samples per record
SIZE_RECORD = 2070  # total size of record (2x1024 B samples + record header)
REC_MARKER = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 255], dtype=np.uint8)


def header_dt(size_header=SIZE_HEADER):
    return np.dtype([('Header', 'S%d' % size_header)])


def data_dt(num_samples=NUM_SAMPLES):
    return np.dtype([('timestamp', np.int64),            # 8 Byte
                    ('n_samples', np.uint16),           # 2 Byte
                    ('rec_num', np.uint16),             # 2 Byte
                    ('samples', ('>i2', num_samples)),  # 2 Byte each x 1024 typ.; note endian type
                    ('rec_mark', (np.uint8, 10))])      # 10 Byte

# data type of .continuous open ephys 0.2x file format header
HEADER_DT = header_dt()

# data type of individual records, n Bytes
DATA_DT = data_dt()  # (2048 + 22) Byte = 2070 Byte total


def read_header(filename):
    """Return dict with .continuous file header content."""
    # TODO: Compare headers, should be identical except for channel

    # 1 kiB header string data type
    header = read_segment(filename, offset=0, count=1, dtype=HEADER_DT)

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


class ContinuousReader(object):
    def __init__(self, file_path, *args, **kwargs):
        self.file_path = os.path.abspath(file_path) 
        assert os.path.isfile(self.file_path)
        self.file_size = os.path.getsize(self.file_path)
        
        self.chunk_size = None
        self.bytes_read = None
        self.file_size = None

        self.buffer = None
        self._header = None
        
        self.header_dt = HEADER_DT 
        self.data_dt = DATA_DT 

    @property
    def header(self):
        if self._header is None:
            self._header = read_header(self.file_path)
        return self._header

    @header.setter
    def header(self, value):
        pass

    def read_header(self):
        """Return dict with .continuous file header content."""
        # TODO: Compare headers, should be identical except for channel

        # 1 kiB header string data type
        header = read_segment(self.file_path, offset=0, count=1, dtype=HEADER_DT)

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

    def read_chunk(self):
        pass

    def check_chunk(self):
        pass

    def check_data(self):
        pass

    def read_segment(self, offset, count, dtype):
        """Read segment of a file from [offset] for [count]x[dtype]"""
        with open(self.file_path, 'rb') as fid:
            fid.seek(offset)
            segment = np.fromfile(fid, dtype=dtype, count=count)
        return segment


class SliceReader(object):
    """ Slice reader represents a single slice through all files in a single directory.
    """
    def __init__(self, *args, **kwargs):
        self.input_directory = None
        assert os.path.isdir(os.path.abspath(self.input_directory))
        self.buffer = None
        self.headers = None
        self.files = None
        self.channels = None
        self.proc_node = None

    def stack_files(self):
        pass

    def gather_files(self):
        base_name = os.path.join(self.input_directory, '{proc_node:d}_CH{channel:d}.continuous')
        file_names = [base_name.format(proc_node=self.proc_node, channel=chan) for chan in self.channels]
        for f in file_names:
            assert os.path.isfile(f)

        # all input files in a single directory should have equal length
        file_sizes = [os.path.getsize(fname) for fname in file_names]
        assert len(set(file_sizes)) == 1

        return file_names, file_sizes[0]
        

def gather_files(input_directory, channels, proc_node):
    """Return list of paths to valid input files for the input directory."""
    #TODO: Handling non-existing items in file list (i.e. directories named like .continuous files)

    base_name = os.path.join(input_directory, '{proc_node:d}_CH{channel:d}.continuous')
    file_names = [base_name.format(proc_node=proc_node, channel=chan) for chan in channels]
    for f in file_names:
        #print 'Is {file} an existing file? {existence}'.format(file=f, existence='YES' if os.path.isfile(f) else 'NO!')
        assert os.path.isfile(f)

    # all input files in a single directory should have equal length
    file_sizes = [os.path.getsize(fname) for fname in file_names]
    assert len(set(file_sizes)) == 1

    return file_names, file_sizes[0]


def read_segment(filename, offset, count, dtype):
    """Read segment of a file from [offset] for [count]x[dtype]"""
    with open(filename, 'rb') as fid:
        fid.seek(offset)
        segment = np.fromfile(fid, dtype=dtype, count=count)
    return segment


def read_records(filename, record_offset=0, record_count=10000,
                 size_header=SIZE_HEADER, num_samples=NUM_SAMPLES, size_record=SIZE_RECORD):
    return read_segment(filename, offset=size_header + record_offset*size_record, count=record_count, dtype=DATA_DT)


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


def check_inputs(input_directories):
    """Check input directories for:
    - existence of all required files
    - matching files in all input directories
    - equal size of all files in a single directory
    - matching sampling rates
    - matching file size given record and header sizes
    - check for trailing zeros (check last for all 0, check at 2*distance each, then half distance to slow down)

    Args:
        List of input directories

    Returns:
        True if correct, False if errors occurred
    """
    # TODO: You know... do stuff.
    print input_directories


def data_to_buffer(file_paths, count=1000, buf=None, size_record=SIZE_RECORD):
    """Read [count] records from [proc_node] file at [filepath] into a buffer."""
    assert channel_list is Iterable
    
    # temporary storage
    buf = buf if buf is not None else np.zeros((len(channel_list), count * size_record), dtype='>i2')

    return buf


def folder_to_dat(in_dir, outfile, channels, proc_node=100, append=False, chunk_size=100000,
                  size_record=SIZE_RECORD, size_header=SIZE_HEADER, *args, **kwargs):
    """Given an input directory [in_dir] will write or append .continuous files for channels in [channels] iterable
    of an open ephys [proc_node] into single [outfile] .dat file.
    Data is transferred in batches of [chunk_size] records per channel.
    """
    # TODO: out_log file should be ast.eval()-uable (i.e. exported as dict)
    # TODO: use logging module for event logging?

    start_t = time.time()

    log__string_input = '======== {in_dir} ========\n' + \
                        'Channels: {channels}, proc_node: {proc_node}, write mode: {file_mode}\n\n'
    log_string_item = '--> Reading "{filename}"\nHeader = {header_str}\n\n'
    log_string_chunk = '  ~ Reading {count} records ({start}:{end}) from "{filename}"\n'
    file_mode = 'a' if append else 'w'

    file_paths, file_sizes = gather_files(in_dir, channels, proc_node)
    assert len(file_paths) == len(channels)

    # preallocate temporary storage
    buf = np.zeros((len(channels), chunk_size*size_record), dtype='>i2')

    # calculate number of records from file size
    # there should be an integer multiple of records, i.e. no leftover bytes!
    assert not (file_sizes-size_header) % size_record
    num_records = (file_sizes-size_header) // size_record
    records_left = num_records
    bytes_written = 0

    try:
        with open(outfile, file_mode+'b') as out_fid_dat, open(outfile+'.log', file_mode) as out_fid_log:
            out_fid_log.write(log__string_input.format(in_dir=in_dir, channels=str(channels),
                                                       proc_node=proc_node, file_mode=file_mode))
            for fname in file_paths:
                out_fid_log.write(log_string_item.format(filename=fname, header_str=str(read_header(fname))))

            # loop over all records, in chunk sizes
            while records_left:
                count = min(records_left, chunk_size)
                offset = num_records - records_left

                # load chunk of data from all channels
                for n, fname in enumerate(file_paths):
                    buf[n, 0:count*NUM_SAMPLES] = read_records(fname, record_count=count,
                                                               record_offset=offset)['samples'].ravel()
                    out_fid_log.write(log_string_chunk.format(filename=fname, count=count,
                                                              start=offset, end=offset+count-1))

                # write chunk of interleaved data
                if count == chunk_size:
                    buf.transpose().tofile(out_fid_dat)
                else:
                    # We don't want to write the trailing zeros on the last chunk
                    buf[:, 0:count*NUM_SAMPLES].transpose().tofile(out_fid_dat)
                    out_fid_log.write('\n')

                records_left -= count
                bytes_written += (count * 2048 * len(channels))  # buf[:, 0:count*NUM_SAMPLES].nbytes

            elapsed = time.time() - start_t
            bytes_written /= 1e6
            speed = bytes_written/elapsed
            msg_str = '\nDone! Stacked '+'' if not append else 'another' + \
                      '{0:.2f} MB from {channels} channels into "{1:s}", took {2:.2f} s, effectively {3:.2f} MB/s'

            msg_str = msg_str.format(bytes_written, os.path.abspath(outfile), elapsed, speed, channels=len(channels))
            print msg_str
            out_fid_log.write(msg_str+'\n\n')

    except IOError as e:
        print 'Operation failed: {error}'.format(error=e.strerror)


if __name__ == "__main__":

    # Command line interface
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("target", nargs='*',
                        help="""Path/list of paths to directories containing raw .continuous data OR path to .session
                                definition file.
                                Listing multiple files will result in data sets being concatenated in listed order.""")

    channel_group = parser.add_mutually_exclusive_group()

    channel_group.add_argument("-c", "--channel_list", nargs='*', default=[], type=int,
                               help="""List of channels to merge from the .probe file""")

    channel_group.add_argument("-C", "--channel-count", default=64, type=int,
                               help="Number of consecutive channels, 1-based index. E.g. 64: Channels 1:64")

    parser.add_argument("-l", "--layout", help="Path to .probe file.")
    parser.add_argument("-p", "--params", help="Path to .params file.")
    parser.add_argument("-o", "--output", default='raw.dat', help="Output file path.")
    parser.add_argument("--remove_trailing_zeros", action='store_true')
    cli_args = parser.parse_args()

    channel_list = cli_args.channel_list if cli_args.channel_list else list(range(1, cli_args.channel_count+1))

    if cli_args.remove_trailing_zeros:
        raise NotImplementedError("Can't remove trailing zeros just yet.")

    for append_dat, directory in enumerate(cli_args.target):
        folder_to_dat(in_dir=directory,
                      outfile=cli_args.output,
                      channels=channel_list,
                      append=bool(append_dat))
