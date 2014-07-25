#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Jul 25, 2014 04:50
@author: <'Ronny Eichler'> ronny.eichler@gmail.com

Generator tests...
"""

import os
import time
import numpy as np

SIZE_HEADER = 1024  # size of header in B
NUM_SAMPLES = 1024  # number of samples per record
SIZE_RECORD = 2070  # total size of record (2x1024 B samples + record header)
REC_MARKER = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 255], dtype=np.uint8)

# data type of .continuous open ephys 0.2x file format header
HEADER_DT = np.dtype([('Header', 'S%d' % SIZE_HEADER)])

# (2048 + 22) Byte = 2070 Byte total
DATA_DT = np.dtype([('timestamp', np.int64),            # 8 Byte
                    ('n_samples', np.uint16),           # 2 Byte
                    ('rec_num', np.uint16),             # 2 Byte
                    ('samples', ('>i2', NUM_SAMPLES)),  # 2 Byte each x 1024 typ.; note endian type
                    ('rec_mark', (np.uint8, 10))])      # 10 Byte


def reader(filename, buf):
    with open(filename, 'rb') as fid:
        yield np.fromfile(fid, HEADER_DT, 1)
        while True:
            data = np.fromfile(fid, DATA_DT, len(buf)/NUM_SAMPLES)
            buf[:len(data)*NUM_SAMPLES] = data['samples'].ravel()
            yield {idx: data[idx] for idx in data.dtype.names if idx != 'samples'} if len(data) else None


def folder_to_dat(in_dir_template, out_file, channels, filemode='w', chunk_size=1000000, proc_node=100, *args, **kwargs):
    start_t, bytes_written = time.time(), 0
    buf = np.zeros((len(channels), chunk_size*NUM_SAMPLES), dtype='>i2')
    readers = [reader(in_dir_template.format(proc_node, chan), buf[n, :]) for n, chan in enumerate(channels)]
    headers = [r.next() for r in readers]
    with open(out_file, filemode) as out_fid_dat:
        while None not in [r.next() for r in readers]:
            buf.tofile(out_fid_dat)
            bytes_written += buf.nbytes
    elapsed = time.time()-start_t
    print '{0:02f} s, {1:02f} MB/s'.format(elapsed, (bytes_written/10**6)/elapsed)

if __name__ == "__main__":
    # Command line interface
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("target", nargs='*', help='List of target directories')
    channel_group = parser.add_mutually_exclusive_group()
    channel_group.add_argument("-c", "--channel_list", nargs='*', default=[], type=int, help="List of channels")
    channel_group.add_argument("-C", "--channel-count", default=64, type=int, help="Number of consecutive channels")
    parser.add_argument("-o", "--output", default='raw.dat', help="Output file path.")
    cli_args = parser.parse_args()
    channel_list = cli_args.channel_list if cli_args.channel_list else list(range(1, cli_args.channel_count+1))
    for append_dat, directory in enumerate(cli_args.target):
        folder_to_dat(in_dir_template=os.path.join(directory, '{0:d}_CH{1:d}.continuous'),
                      out_file=cli_args.output,
                      channels=channel_list,
                      filemode='ab' if bool(append_dat) else 'wb',
                      chunk_size=100)



# class IterableThingy(object):
#     def __init__(self):
#         self.generator = self.test_gen()
#
#     @staticmethod
#     def test_gen():
#         print "First call!"
#         try:
#             with open('testfile.txt', 'r') as fid:
#                     while True:
#                         rv = fid.readline().strip()
#                         if len(rv):
#                             offset = (yield rv)  # this can give a new offset to the generator via "send()"
#                             if offset:
#                                 print "New offset:", offset
#                         else:
#                             raise StopIteration('No more text!')
#         finally:
#             print "Closing call!"
#
#     def __iter__(self):
#         return self
#
#     def next(self):
#         return self.generator.next()
#         # if rv is None:
#         #     self.generator.close()
#         # else:
#         #     return rv
#
#     def __getitem__(self, item):
#         return self.generator.send(item)
#
# def main():
#     ir = IterableThingy()
#     for rv in ir:
#         if '2' in rv:
#             ir[2]
#         print rv