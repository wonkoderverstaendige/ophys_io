#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# import sys
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


def get_needed_channels(cli_args=None):
    """
    gets a list of channels that are needed in order to process a given channel
    """
    if cli_args is None:
        import argparse
        parser = argparse.ArgumentParser(description=get_needed_channels.__doc__)
        parser.add_argument('probe_file', nargs=1,
                            help="""the probe file to be used""")
        parser.add_argument("groups", nargs='+',
                            help="""a list of groups""")
        parser.add_argument("-f", "--filenames", action='store_true',
                            help="""returns a list of open-ephys continuous filenames, instead of a list of channel
                                    numbers""")
        parser.add_argument("-n", "--node", type=int,
                            help="""a node number for the filenames (default 100)""")
        parser.add_argument("--zerobased", action='store_true',
                            help="use klusta zero-based convention instead of open-ephys 1-based one")
        cli_args = parser.parse_args()

    probe_file = cli_args.probe_file[0]
    groups = [int(g) for g in cli_args.groups]

    do_filenames = False
    if cli_args.filenames:
        do_filenames = True

    if cli_args.node:
        node = cli_args.node
        do_filenames = True
    else:
        node = 100

    zero_based = False
    if cli_args.zerobased:
        zero_based = True

    layout = util.run_prb(probe_file)

    chans = []
    for g in groups:
        chans.extend(layout['channel_groups'][g]['channels'])
        if 'reference' in layout['channel_groups']:
            chans.extend(layout['channel_groups'][g]['reference'])

    if 'ref_a' in layout:
        chans.extend(layout['ref_a'])

    if 'ref_b' in layout:
        chans.extend(layout['ref_b'])


    if not zero_based:
        chans = [c + 1 for c in chans]
    chans = set(chans)

    if do_filenames:
        filenames = [str(node) + '_CH' + str(c) + '.continuous' for c in chans]
        print('\n'.join(filenames))
    else:
        print(' '.join(map(str, chans)))


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
        parser.add_argument("-S", "--split-groups", action="store_true",
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
        parser.add_argument("-g", "--channel-groups", type=int, nargs="+",
                            help="limit to only a subset of the channel groups")
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
        channel_groups = {0: {'channels': list(range(cli_args.channel_count))}}
    elif cli_args.channel_list is not None:
        channel_groups = {0: {'channels': cli_args.channel_list}}
    elif cli_args.layout is not None:
        layout = util.run_prb(cli_args.layout)
        if cli_args.split_groups:
            channel_groups = layout['channel_groups']
            dead_channels = layout['dead_channels'] if 'dead_channels' in layout else []
            if cli_args.channel_groups:
                channel_groups = {i: channel_groups[i] for i in cli_args.channel_groups if i in channel_groups}
        else:
            channels, dead_channels = util.flat_channel_list(layout)
            logging.warning('Not splitting groups! {}, ')
            # make a new channel group by merging in the existing ones
            channel_groups = {0: {'channels': channels,
                                  'dead_channels': dead_channels}}
    else:
        warnings.warn('No channels given, using all found in target directory.')
        raise NotImplementedError('Grabbing all channels from file names not done yet. Sorry.')

    prm_in_file = None
    if cli_args.params is not None:
        prm_in_file = cli_args.params


    # involved file names
    if cli_args.output is None:
        out_path, out_file, out_ext = '', op.basename(op.splitext(cli_args.target[0])[0]), "dat"
    else:
        out_path, out_file = op.split(op.expanduser(cli_args.output))
        if out_file == '':
            out_file = op.basename(cli_args.target[0])

        out_file, out_ext = op.splitext(out_file)
        out_ext = out_ext.strip('.')
        if out_ext == '':
            out_ext = 'dat'

    if len(out_path) and not op.exists(out_path):
        os.mkdir(out_path)
    # indicate when more than one source was merged into the .dat file
    out_file += "+{}files".format(len(cli_args.target) - 1) if len(cli_args.target) > 1 else ''

    logging.debug('Output path, file, extension: "{}", "{}", "{}"'.format(out_path, out_file, out_ext))
    logging.info(cli_args.zero_dead_channels)

    time_written = 0
    for cg_id, channel_group in channel_groups.items():
        logging.debug('channel group: {}'.format(channel_group))

        crs = util.fmt_channel_ranges(channel_group['channels'])
        output_base_name = "{outfile}--cg({cg_id:02})_ch[{crs}]".format(outfile=out_file, cg_id=cg_id, crs=crs)
        output_file_name = '.'.join([output_base_name, out_ext])
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
            if prm_in_file:
                f = open(prm_in_file, 'r')
                prm_in = f.read()
                f.close()
            else:
                prm_in = pkgr.resource_string('config', 'default.prm').decode()
            prm_out.write(prm_in.format(experiment_name=output_base_name,
                                        probe_file=output_base_name + '.prb',
                                        raw_file=output_file_path,
                                        n_channels=len(channel_group['channels'])))

    logging.info('Done! Total data duration: {}'.format(util.fmt_time(time_written)))


if __name__ == "__main__":
    main()
