#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Jun 15, 2014 23:40
@author: <'Ronny Eichler'> ronny.eichler@gmail.com

Reads Open Ephys .continuous files and converts them to raw 16-bit .dat files
"""

import os
import time
import open_ephys_io as oe

LOG_STR_INPUT = '======== {in_dir} ========\n' + \
                    'Channels: {channels}, reference: {reference}, ' \
                    'proc_node: {proc_node}, write mode: {file_mode}\n\n'
LOG_STR_ITEM = '--> Reading "{filename}"\nHeader = {header_str}\n\n'
LOG_STR_CHUNK = '  ~ Reading {count} records ({start}:{end}) from "{filename}"\n'


def continuous_to_dat(input_path, output_path, channel_group, proc_node=100, append=False, chunk_size=1e3,
                      limit_dur=False):
    """Given an input directory [in_dir] will write or append .continuous files for channels in [channels] iterable
    of an open ephys [proc_node] into single [outfile] .dat file.
    Data is transferred in batches of [chunk_size] records per channel.
    """
    # TODO: use logging module for event logging

    start_t = time.time()

    file_mode = 'a' if append else 'w'

    data_channels = [cid + 1 for cid in channel_group['channels']]
    ref_channels = [rid + 1 for rid in channel_group['reference']] if "reference" in channel_group else None

    file_paths, file_sizes = oe.gather_files(input_path, data_channels, proc_node)

    assert len(file_paths) == len(data_channels)

    # preallocate temporary storage
    buf = oe.make_buffer(len(data_channels), chunk_size)

    # calculate number of records from file size
    # there should be an integer multiple of records, i.e. no leftover bytes!
    assert not (file_sizes - oe.SIZE_HEADER) % oe.SIZE_RECORD
    num_records = (file_sizes - oe.SIZE_HEADER) // oe.SIZE_RECORD

    # FIXME: Get sampling rate from header!
    # If duration limited, find max number of records that should be grabbed
    records_left = num_records if not limit_dur else min(num_records, int(float(limit_dur) * 30000 / oe.NUM_SAMPLES))
    bytes_written = 0

    try:
        with open(output_path, file_mode + 'b') as out_fid_dat,\
             open(output_path + '.log', file_mode) as out_fid_log:

            out_fid_log.write(LOG_STR_INPUT.format(in_dir=input_path, channels=str(channel_group),
                                                   reference=ref_channels, proc_node=proc_node,
                                                   file_mode=file_mode))
            for fname in file_paths:
                out_fid_log.write(LOG_STR_ITEM.format(filename=fname, header_str=str(oe.read_header(fname))))

            # loop over all records, in chunk sizes
            while records_left:
                count = min(records_left, chunk_size)
                offset = num_records - records_left

                # load chunk of data from all channels
                for n, fname in enumerate(file_paths):
                    buf[n, 0:count * oe.NUM_SAMPLES] = oe.read_records(fname, record_count=count,
                                                                       record_offset=offset)['samples'].ravel()
                    out_fid_log.write(LOG_STR_CHUNK.format(filename=fname, count=count,
                                                           start=offset, end=offset + count - 1))

                # write chunk of interleaved data
                if count == chunk_size:
                    buf.transpose().tofile(out_fid_dat)
                else:
                    # We don't want to write the trailing zeros on the last chunk
                    buf[:, 0:count * oe.NUM_SAMPLES].transpose().tofile(out_fid_dat)
                    out_fid_log.write('\n')

                records_left -= count
                bytes_written += (count * 2048 * len(data_channels))  # buf[:, 0:count*NUM_SAMPLES].nbytes

            elapsed = time.time() - start_t
            bytes_written /= 1e6
            speed = bytes_written / elapsed
            msg_str = '{appended} {channels} channels into "{1:s}"\n' \
                      '({0:.2f} MB in {2:.2f} s, effectively {3:.2f} MB/s)\n'

            msg_str = msg_str.format(bytes_written, os.path.abspath(output_path), elapsed, speed,
                                     channels=len(data_channels), appended="Appended" if append else "Wrote")
            print(msg_str)
            out_fid_log.write(msg_str + '\n\n')

    except IOError as e:
        print('Operation failed: {error}'.format(error=e.strerror))


def kwik_to_dat(*args, **kwargs):
    raise NotImplementedError
