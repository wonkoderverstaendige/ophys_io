#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Jun 15, 2014 23:40
@author: <'Ronny Eichler'> ronny.eichler@gmail.com

Reads Open Ephys .continuous files and converts them to raw 16-bit .dat files
"""

import os
import time
from contextlib import ExitStack

import numpy as np
import oio.open_ephys_io as oe
from oio.util import fmt_time

LOG_STR_INPUT = '======== {in_dir} ========\n' + \
                'Channels: {channels}, reference: {reference}, ' \
                'proc_node: {proc_node}, write mode: {file_mode}\n\n'
LOG_STR_ITEM = '--> Reading "{filename}"\nHeader = {header_str}\n\n'
LOG_STR_CHUNK = '  ~ Reading {count} records ({start}:{end}) from "{filename}"\n'


def continuous_to_dat(input_path, output_path, channel_group, proc_node=100,
                      append=False, chunk_records=10000, limit_dur=False, dead_channels=None):
    """Given an input directory [in_dir] will write or append .continuous files for channels in [channels] iterable
    of an open ephys [proc_node] into single [outfile] .dat file.
    Data is transferred in batches of [chunk_records] 2kiB records per channel.
    """
    # TODO: use logging module for event logging
    # TODO: Referencing

    start_t = time.time()
    file_mode = 'a' if append else 'w'

    # NOTE: Channel numbers zero-based in configuration, but not in file name space. Grml.
    data_channels = [cid + 1 for cid in channel_group['channels']]
    ref_channels = [rid + 1 for rid in channel_group['reference']] if "reference" in channel_group else []
    dead_channels = [did + 1 for did in dead_channels]

    dead_channels_indices = [data_channels.index(dc) for dc in dead_channels if dc in data_channels]

    data_file_paths = oe.gather_files(input_path, data_channels, proc_node)
    ref_file_paths = oe.gather_files(input_path, ref_channels, proc_node)

    try:
        with ExitStack() as stack, open(output_path, file_mode + 'b') as out_fid_dat, \
                open(output_path + '.log', file_mode) as out_fid_log:

            out_fid_log.write(LOG_STR_INPUT.format(in_dir=input_path, channels=str(channel_group),
                                                   reference=ref_channels, proc_node=proc_node,
                                                   file_mode=file_mode))

            data_files = [stack.enter_context(oe.ContinuousFile(f)) for f in data_file_paths]
            ref_files = [stack.enter_context(oe.ContinuousFile(f)) for f in ref_file_paths]

            num_records, sampling_rate, buffer_size, block_size = oe.check_headers(data_files + ref_files)
            for f in data_files:
                out_fid_log.write(LOG_STR_ITEM.format(filename=f.path, header_str=str(f.header)))

            # If duration limited, find max number of records that should be grabbed
            records_left = num_records if not limit_dur \
                else min(num_records, int(float(limit_dur) * sampling_rate / block_size))

            # # preallocate temporary storage
            # buf = oe.make_buffer(len(data_channels), chunk_records)

            # loop over all records, in chunk sizes
            bytes_written = 0
            while records_left:
                # print(num_records, records_left)
                count = min(records_left, chunk_records)

                res = np.vstack([f.read_record(count) for f in data_files])

                # reference channels if needed
                if len(ref_channels):
                    res -= np.vstack([f.read_record(count) for f in ref_files]).mean(axis=0, dtype=np.int16)

                # zero dead channels if needed
                if len(dead_channels_indices):
                    zeros = np.zeros_like(res[0])
                    for dci in dead_channels_indices:
                        print("zero-ing channel {}".format(data_channels[dci]))
                        res[dci] = zeros

                res.transpose().tofile(out_fid_dat)

                # offset = num_records - records_left

                # # load chunk of data from all channels
                # for n, fname in enumerate(file_paths):
                #     buf[n, 0:count * oe.NUM_SAMPLES] = oe.read_records(fname, record_count=count,
                #                                                        record_offset=offset)['samples'].ravel()
                #     out_fid_log.write(LOG_STR_CHUNK.format(filename=fname, count=count,
                #                                            start=offset, end=offset + count - 1))
                #
                #     # write chunk of interleaved data
                #     if count == chunk_records:
                #         buf.transpose().tofile(out_fid_dat)
                #     else:
                #         # We don't want to write the trailing zeros on the last chunk
                #         buf[:, 0:count * oe.NUM_SAMPLES].transpose().tofile(out_fid_dat)
                #         out_fid_log.write('\n')

                records_left -= count
                bytes_written += (count * 2048 * len(data_channels))

            duration = bytes_written/(2*sampling_rate)
            elapsed = time.time() - start_t
            speed = bytes_written / elapsed
            msg_str = '{appended} {channels} channels into "{op:s}"\n' \
                      '({dur:s} [{bw:.2f} MB] in {et:.2f} s [{ts:.2f} MB/s])\n'

            msg_str = msg_str.format(appended="Appended" if append else "Wrote", channels=len(data_channels),
                                     op=os.path.abspath(output_path),
                                     dur=fmt_time(duration), bw=bytes_written/1e6, et=elapsed, ts=speed/1e6)
            print(msg_str)
            out_fid_log.write(msg_str + '\n\n')

    except IOError as e:
        print('Operation failed: {error}'.format(error=e.strerror))


def kwik_to_dat(*args, **kwargs):
    raise NotImplementedError
