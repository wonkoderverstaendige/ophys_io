#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import warnings
import logging
import os
import os.path as op
import subprocess
import pkg_resources as pkgr
import pprint

from oio import util
from oio.conversion import continuous_to_dat

try:
    version = subprocess.check_output(["git", "describe", "--always"]).strip().decode('utf-8')
except:
    version = "Unknown"

WRITE_DATA = True


def main(cli_args=None):
    # Command line interface
    if cli_args is None:
        import argparse

        parser = argparse.ArgumentParser()
        parser.add_argument("target", nargs='*',
                            help="""Path/list of paths to directories containing raw .continuous data OR path
                                    to .session definition file. Listing multiple files will result in data sets
                                    being concatenated in listed order.""")
        parser.add_argument('-v', '--verbose', action='store_true',
                            help="Verbose (debug) output")

        channel_group = parser.add_mutually_exclusive_group()
        channel_group.add_argument("-c", "--channel-count", type=int,
                                   help="Number of consecutive channels, 1-based index. E.g. 64: Channels 1:64")
        channel_group.add_argument("-C", "--channel-list", nargs='*', type=int,
                                   help="List of channels in order they are to be merged.")
        channel_group.add_argument("-l", "--layout",
                                   help="Path to klusta .probe file.")

        parser.add_argument("-d", "--dead-channels", nargs='*', type=int,
                            help="List of dead channels. If flag set, these will be set to zero.")
        parser.add_argument("-S", "--split-groups", action="store_false",
                            help="Split channel groups into separate files.")
        parser.add_argument("-n", "--proc-node", help="Processor node id", type=int, default=100)
        parser.add_argument("-p", "--params", help="Path to .params file.")
        parser.add_argument("-o", "--output", default=None, help="Output file path. Name of target if none given.")
        parser.add_argument("-D", "--duration", type=int, help="Maximum  duration of recording (s)")
        parser.add_argument("--chunk-records", default=10000, type=int,
                            help="Number of records (2 kiB/channel) to read at a time. Increase to speed up, reduce to"
                                 "deal with memory limitations. Default: ~20 MiB/channel")
        parser.add_argument("--remove-trailing-zeros", action='store_true')
        parser.add_argument("--dry-run", action='store_true')
        parser.add_argument("-z", "--zero-dead-channels", action='store_true')
        cli_args = parser.parse_args()

    log_level = logging.DEBUG if cli_args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format='%(asctime)s %(message)s')
    logging.info('Starting conversion with OIO git version {}'.format(version))

    if cli_args.remove_trailing_zeros:
        raise NotImplementedError("Can't remove trailing zeros just yet.")
    if cli_args.split_groups:
        # i.e. use the -S flag pretty please and fix this later. Some day. Mayhaps-ish.
        raise NotImplementedError("Explicit channel group merging not supported. Defaulting to split groups.")

    # Set up channel layout (channels, references, dead channels) from command line inputs or layout file
    dead_channels = cli_args.dead_channels if cli_args.dead_channels is not None else []
    if cli_args.channel_count is not None:
        channel_groups = {0: {'channels': list(range(1, cli_args.channel_count + 1))}}
    elif cli_args.channel_list is not None:
        channel_groups = {0: {'channels': cli_args.channel_list}}
    elif cli_args.layout is not None:
        layout = util.run_prb(cli_args.layout)
        channel_groups = layout['channel_groups']
        dead_channels = layout['dead_channels'] if 'dead_channels' in layout else []
    else:
        warnings.warn('No channels given, using all found in target directory.')
        raise NotImplementedError('Grabbing all channels from file names not done yet. Sorry.')

    # involved file names
    if cli_args.output is None:
        out_path, out_file, output_ext = '', op.basename(op.splitext(cli_args.target[0])[0]), "dat"
    else:
        out_path, out_file = op.split(op.expanduser(cli_args.output))
        if out_file == '':
            out_file = op.basename(cli_args.target[0])
            output_ext = "dat"
        else:
            output_ext = op.splitext(out_file)

    if not op.exists(out_path):
        os.mkdir(out_path)
    # indicate when more than one source was merged into the .dat file
    out_file += "+{}files".format(len(cli_args.target) - 1) if len(cli_args.target) > 1 else ''

    time_written = 0
    for cg_id, channel_group in channel_groups.items():
        crs = util.fmt_channel_ranges(channel_group['channels'])
        output_base_name = "{outfile}--cg({cg_id:02})_ch[{crs}]".format(outfile=out_file, cg_id=cg_id, crs=crs)
        output_file_name = '.'.join([output_base_name, output_ext])
        output_file_path = op.join(out_path, output_file_name)

        time_written = 0
        for file_mode, input_file_path in enumerate(cli_args.target):
            duration = None if cli_args.duration is None else cli_args.duration - time_written

            if not cli_args.dry_run and WRITE_DATA:
                time_written += continuous_to_dat(
                    input_path=op.expanduser(input_file_path),
                    output_path=output_file_path,
                    channel_group=channel_group,
                    dead_channels=dead_channels,
                    zero_dead_channels=cli_args.zero_dead_channels,
                    proc_node=cli_args.proc_node,
                    file_mode='a' if file_mode else 'w',
                    chunk_records=cli_args.chunk_records,
                    duration=duration)

        # create the per-group .prb and .prm files
        with open(op.join(out_path, output_base_name + '.prb'), 'w') as prb_out:
            cg_dict = {cg_id: channel_group}
            if cli_args.zero_dead_channels:
                cg_dict[cg_id]['dead_channels'] = [dc for dc in dead_channels if dc in channel_group['channels']]
            prb_out.write('channel_groups = {}'.format(pprint.pformat(cg_dict)))

        with open(op.join(out_path, output_base_name + '.prm'), 'w') as prm_out:
            prm_in = pkgr.resource_string('config', 'default.prm').decode()
            prm_out.write(prm_in.format(experiment_name=output_base_name,
                                        probe_file=output_base_name + '.prb',
                                        raw_file=output_file_path,
                                        n_channels=len(channel_group['channels'])))

    logging.info('Done! Total data duration: {}'.format(util.fmt_time(time_written)))


if __name__ == "__main__":
    main()
