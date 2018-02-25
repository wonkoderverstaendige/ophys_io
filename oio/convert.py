#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Jun 15, 2014 23:40
@author: <'Ronny Eichler'> ronny.eichler@gmail.com

Reads Open Ephys .continuous files and converts them to raw 16-bit .dat files
"""

import os
import time
import logging
import os.path as op
from contextlib import ExitStack

import numpy as np
import oio.open_ephys_io_deprecated as oe
from oio.util import fmt_time

LOG_STR_INPUT = '==> Input: {path}'
LOG_STR_OUTPUT = '<== Output {path}'
LOG_STR_CHAN = 'Channels: {channels}, reference: {reference}, Dead: {dead}, ' \
               'proc_node: {proc_node}, write mode: {file_mode}'
LOG_STR_ITEM = ', Header: channel: {header[channel]}, date: {header[date_created]}'
DEBUG_STR_CHUNK = '~ Reading {count} records (left: {left}, max: {num_records})'
DEBUG_STR_REREF = '~ Re-referencing by subtracting average of channels {channels}'
DEBUG_STR_ZEROS = '~ Zeroing (Flag: {flag}) dead channel {channel}'

MODE_STR = {'a': 'Append', 'w': "Write"}
MODE_STR_PAST = {'a': 'Appended', 'w': "Wrote"}


def continuous_to_dat(input_path, output_path, channel_group, proc_node=100,
                      file_mode='w', chunk_records=10000, duration=False,
                      dead_channels=None, zero_dead_channels=True):
    """Given an input directory [in_dir] will write or append .continuous files for channels in [channels] iterable
    of an open ephys [proc_node] into single [outfile] .dat file.
    Data is transferred in batches of [chunk_records] 2kiB records per channel.
    """
    start_t = time.time()
    logger = logging.getLogger(output_path)
    file_handler = logging.FileHandler(output_path + '.log', mode=file_mode)
    formatter = logging.Formatter('%(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.setLevel(logging.DEBUG)

    logger.info(LOG_STR_INPUT.format(path=input_path))
    logger.info(LOG_STR_OUTPUT.format(path=output_path))

    # NOTE: Channel numbers zero-based in configuration, but not in file name space. Grml.
    data_channels = [cid + 1 for cid in channel_group['channels']]
    ref_channels = [rid + 1 for rid in channel_group['reference']] if "reference" in channel_group else []
    dead_channels = [did + 1 for did in dead_channels]
    logger.info("Dead channels to zero: {}, {}".format(zero_dead_channels, dead_channels))

    dead_channels_indices = [data_channels.index(dc) for dc in dead_channels if dc in data_channels]

    data_file_paths = oe.gather_files(input_path, data_channels, proc_node)
    ref_file_paths = oe.gather_files(input_path, ref_channels, proc_node)

    logger.info(LOG_STR_CHAN.format(channels=data_channels,
                                    reference=ref_channels, dead=dead_channels,
                                    proc_node=proc_node, file_mode=MODE_STR[file_mode]))

    try:
        with ExitStack() as stack, open(output_path, file_mode + 'b') as out_fid_dat:

            data_files = [stack.enter_context(oe.ContinuousFile(f)) for f in data_file_paths]
            ref_files = [stack.enter_context(oe.ContinuousFile(f)) for f in ref_file_paths]
            for oe_file in data_files:
                logger.info("Open data file: {}".format(op.basename(oe_file.path)) +
                            LOG_STR_ITEM.format(header=oe_file.header))
            for oe_file in ref_files:
                logger.info("Open reference file: {}".format(op.basename(oe_file.path)) +
                            LOG_STR_ITEM.format(header=oe_file.header))

            num_records, sampling_rate, buffer_size, block_size = oe.check_headers(data_files + ref_files)

            # If duration limited, find max number of records that should be grabbed
            records_left = num_records if not duration \
                else min(num_records, int(duration * sampling_rate // block_size))
            if records_left < 1:
                epsilon = 1/sampling_rate*block_size*1000
                logger.warning("Remaining duration limit ({:.0f} ms) less than duration of single block ({:.0f} ms)."
                               " Skipping target.".format(duration*1000, epsilon))
                return 0

            # loop over all records, in chunk sizes
            bytes_written = 0
            while records_left:
                count = min(records_left, chunk_records)

                logger.debug(DEBUG_STR_CHUNK.format(count=count, left=records_left,
                             num_records=num_records))
                res = np.vstack([f.read_record(count) for f in data_files])

                # reference channels if needed
                if len(ref_channels):
                    logger.debug(DEBUG_STR_REREF.format(channels=ref_channels))
                    res -= np.vstack([f.read_record(count) for f in ref_files]).mean(axis=0, dtype=np.int16)

                # zero dead channels if needed
                if len(dead_channels_indices) and zero_dead_channels:
                    zeros = np.zeros_like(res[0])
                    for dci in dead_channels_indices:
                        logger.debug(DEBUG_STR_ZEROS.format(flag=zero_dead_channels, channel=data_channels[dci]))
                        res[dci] = zeros

                res.transpose().tofile(out_fid_dat)

                records_left -= count
                bytes_written += (count * 2048 * len(data_channels))

            data_duration = bytes_written / (2 * sampling_rate * len(data_channels))
            elapsed = time.time() - start_t
            speed = bytes_written / elapsed
            logger.info('{appended} {channels} channels into "{op:s}"'.format(
                appended=MODE_STR_PAST[file_mode], channels=len(data_channels),
                op=os.path.abspath(output_path)))
            logger.info('{rec} blocks ({dur:s}, {bw:.2f} MB) in {et:.2f} s ({ts:.2f} MB/s)'.format(
                rec=num_records-records_left, dur=fmt_time(data_duration),
                bw=bytes_written / 1e6, et=elapsed, ts=speed / 1e6))

            # returning duration of data written, epsilon=1 sample, allows external loop to make proper judgement if
            # going to next target makes sense via comparison. E.g. if time less than one sample short of
            # duration limit.
            logger.removeHandler(file_handler)
            file_handler.close()

            return data_duration

    except IOError as e:
        print('Operation failed: {error}'.format(error=e.strerror))


def kwik_to_dat(*args, **kwargs):
    print(args, kwargs)
    raise NotImplementedError
