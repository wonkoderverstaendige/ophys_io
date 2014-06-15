#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 5/13/14 9:15 PM 2013
@author: <'Ronny Eichler'> ronny.eichler@gmail.com

"""

__revision__ = '0.0.1'

import os
import sys
import time

import random
import numpy as np
try:
    import matplotlib.pyplot as plt
except:
    raise ImportError
import networkx as nx

from collections import Iterable
import re
import pprint


# fixed header describing data
SIZE_HEADER = 1024

# 22 byte record header + 2048 byte samples
SIZE_RECORD = 2070
NUM_SAMPLES = 1024
REC_MARKER = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 255], dtype=np.uint8)


def channel_graph(tetrodes):
    G = nx.Graph()
    for k, t in tetrodes.items():
        G.add_edges_from(t)
    pos = nx.graphviz_layout(G, prog='neato', args='')
    plt.figure(figsize=(6, 6))
    #colors = np.linspace(0.0, 1.0, num=len(tetrodes))
    colors = [random.random() for n in xrange(len(tetrodes))]
    component = nx.connected_component_subgraphs(G)
    for i, g in enumerate(component):
        #c=*nx.number_of_nodes(g) # random color...
        nx.draw(g, pos, node_size=400, node_color=[colors[i]]*nx.number_of_nodes(g),
                vmin=0.0, vmax=1.0, alpha=0.2, with_labels=True)


def read_header(fname):
    """ Return a dict with header content.
    """
    # TODO: Use alternative reading method when fname is a file id. If string, use numpy.fromfile

    # 1 kiB header
    header_dt = np.dtype([('Header', 'S%d' % SIZE_HEADER)])
    header = np.fromfile(fname, dtype=header_dt, count=1)

    # alternative which moves file pointer
    #fid = open(fname, 'rb')
    #header = fid.read(SIZE_HEADER)
    #fid.close()

    # Stand back! I know regex!
    # Annoyingly, there is a newline character missing in the header
    regex = "header\.([\d\w\.\s]{1,}).=.\'*([^\;\']{1,})\'*"
    header_str = str(header[0][0]).rstrip(' ')
    header_dict = {entry[0]: entry[1] for entry in re.compile(regex).findall(header_str)}
    for key in ['bitVolts', 'sampleRate']:
        header_dict[key] = float(header_dict[key])
    for key in ['blockLength', 'bufferSize', 'header_bytes', 'channel']:
        header_dict[key] = int(header_dict[key]) if not key == 'channel' else int(header_dict[key][2:])
    return header_dict


def read_records(fname, offset=0, count=10000):
    with open(fname, 'rb') as fid:
        # move pointer to new position
        fid.seek(SIZE_HEADER + offset * SIZE_RECORD)

        # n times x B data
        data_dt = np.dtype([('timestamp', np.int64),
                            ('n_samples', np.uint16),
                            ('rec_num', np.uint16),
                            # note endian type
                            ('samples', ('>i2', NUM_SAMPLES)),
                            ('rec_mark', (np.uint8, 10))])

        data = np.fromfile(fid, dtype=data_dt, count=count)
    return data


def check_data(data):
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


def data_to_buffer(path, channels=64, count=1000, proc_node=100, channel_offset=500, buf=None):
    base_end = '{proc_node:d}_CH{channel:d}.continuous'

    # channels to include either given as number of channels, or as a list
    channels = channels if isinstance(channels, Iterable) else list(range(channels))

    # temporary storage
    buf = np.zeros((len(channels), count*1024), dtype='>i2') if buf is None else buf

    # load chunk of data from all channels
    print 'Reading chunk'
    for n in channels:
        fname = path+base_end.format(proc_node=proc_node, channel=n+1)
        # channel offset is clunky shortcut for plotting
        if channel_offset:
            buf[n] = read_records(fname, count=count)['samples'].ravel() + channel_offset * (n-32)
        else:
            buf[n] = read_records(fname, count=count)['samples'].ravel()

    return buf


def continuous_to_dat(dirname, outfile='./proc/raw.dat', channels=64, proc_node=100, chunk_size=10000):
    # TODO: Check that no samples are getting lost, at start, chunk-borders or file end
    # TODO: Instead of channel list, give channel dict and write to appropriate raw file grouping
    #       This won't require re-reading for each tetrode. Not that it takes that long...
    import os
    start_t = time.time()
    fname_base = dirname + '{proc_node:d}_CH{channel:d}.continuous'

    # should be iterable with channels to include into dat file
    channels = channels if isinstance(channels, Iterable) else list(range(channels))

    # list of files
    fnames = [fname_base.format(proc_node=proc_node, channel=chan+1) for chan in channels]
    #print fnames

    # check that file sizes are equal
    fsizes = [os.path.getsize(fname) for fname in fnames]
    assert len(set(fsizes)) == 1

    # there should be an integer multiple of records, i.e. not leftover bytes!
    assert not (fsizes[0] - 1024) % 2070

    # calculate number of records from file size
    num_records = (fsizes[0] - 1024) / 2070

    # pre-allocate array
    buf = np.zeros((len(channels), chunk_size*1024), dtype='>i2')
    #print 'Allocated {0:.2f} MB buffer'.format(buf.nbytes/1e6)

    with open(outfile, 'wb') as fid:
        # loop over all records, in chunk sizes
        records_left = num_records
        chunk_n = 0
        written = 0
        while records_left:
            count = min(records_left, chunk_size)
            offset = num_records - records_left
            #print '-> Reading chunk {0:d}'.format(chunk_n)
            for i, fname in enumerate(fnames):
                buf[i, 0:count*NUM_SAMPLES] = read_records(fname, count=count, offset=offset)['samples'].ravel()
            records_left -= count

            # write chunk of interleaved data
            #print '<- Writing chunk {0:d}'.format(chunk_n)
            if count == chunk_size:
                buf.transpose().tofile(fid)
            else:
                # We don't want to write the trailing zeros on the last chunk
                buf[:, 0:count*NUM_SAMPLES].transpose().tofile(fid)
            written += (count * 2048 * len(channels))
            chunk_n += 1
            #print '{0:.0f}% completed.'.format((num_records-records_left)*100.0/num_records),

    elapsed = time.time() - start_t
    written /= 1e6
    speed = written/elapsed
    msg_str = '\nConverted {0:.2f} MB into {1:s}, took {2:.2f} s, effectively {3:.2f} MB/s'
    print msg_str.format(written, outfile, elapsed, speed)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("echo")
    args =