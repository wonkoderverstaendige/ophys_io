from six import exec_
import os.path as op
import warnings
from collections import Counter
import pkg_resources as pkgr
from .formats import open_ephys, kwik, dat
import logging

DEFAULT_LIMIT = 0.512  # GB

logger = logging.getLogger(__name__)


def get_batch_size(arr, ram_limit_gb=DEFAULT_LIMIT):
    """Return batch size, number of full batches and remainder size for 2D array."""

    batch_size = int(ram_limit_gb * 1e9 / arr.shape[1] / arr.dtype.itemsize)
    return batch_size, arr.shape[0] // batch_size, arr.shape[0] % batch_size


def detect_format(path):
    """Check if/what known data formats are present at the given path and return the module needed to interact with it.
    
    :param path: 
    :return: 
    """

    formats = [f for f in [fmt.detect(path) for fmt in [open_ephys, dat, kwik]] if f is not None]
    if len(formats) == 1:
        fmt = formats[0]
        if 'DAT' in fmt:
            if fmt == 'DAT-File':
                return dat
        else:
            if 'kwik' in fmt:
                return kwik
            else:
                return open_ephys
    logger.info('Detected format(s) {} not valid.'.format(formats))


def run_prb(path):
    """Execute the .prb probe file and import resulting locals return results as dict.
    Args:
        path: file path to probe file with layout as per klusta probe file specs.

    Returns: Dictionary of channel groups with channel list, geometry and connectivity graph.
    """

    if path is None:
        return

    path = op.realpath(op.expanduser(path))
    assert op.exists(path)
    with open(path, 'r') as prb:
        layout = prb.read()

    metadata = {}
    exec_(layout, {}, metadata)
    metadata = {k.lower(): v for (k, v) in metadata.items()}
    return metadata


def flat_channel_list(prb):
    """Flat lists of all channels and bad channels given a probe dictionary.

    Args:
        prb: Path to probe file.

    Returns:
        Tuple of lists of channels.
    """

    channels = sum([prb['channel_groups'][cg]['channels'] for cg in sorted(prb['channel_groups'])], [])
    dead_channels = prb['dead_channels']

    return channels, dead_channels


def monotonic_prb(prb):
    """Return probe file dict with monotonically increasing channel group and channel numbers."""

    # FIXME: Should otherwise copy any other fields over (unknown fields warning)
    # FIXME: Correct ref like dc to indices
    chan_n = 0
    groups = prb['channel_groups']
    monotonic = {}
    for n, chg in enumerate(groups.keys()):
        monotonic[n] = {'channels': list(range(chan_n, chan_n + len(groups[chg]['channels'])))}
        chan_n += len(groups[chg]['channels'])

    # correct bad channel indices
    if 'dead_channels' in prb.keys():
        fcl, fbc = flat_channel_list(prb)
        dead_channels = sorted([fcl.index(bc) for bc in fbc])
    else:
        dead_channels = []
    return monotonic, dead_channels


def make_prb():
    raise NotImplementedError


def make_prm(dat_path, prb_path, n_channels=4):
    prm_in = pkgr.resource_string('config', 'default.prm').decode()
    #
    # with open('default.prm', 'r') as prm_default:
    #     template = prm_default.read()

    base_name, _ = op.splitext(dat_path)
    with open(base_name + '.prm', 'w') as prm_out:
        prm_out.write(prm_in.format(experiment_name=base_name,
                                    probe_file=prb_path,
                                    n_channels=4))


def has_prb(path):
    """Check if file at path has a an accompanying .prb file with the same basename.
    
    Args:
        path: Path to file of interest
        
    Returns:
        Path to .prb file if exists, else None
    """

    base_path, _ = op.splitext(op.abspath(op.expanduser(path)))
    probe_path = base_path + '.prb'
    if op.exists(probe_path) and op.isfile(probe_path):
        return probe_path


def channel_ranges(channel_list):
    """List of channels in to ranges of consecutive channels.

    Args:
        channel_list: list of channel numbers (ints)

    Returns:
        List of list of channels grouped into consecutive sequences.

    Example: channel_ranges([1, 3, 4, 5]) -> [[1], [3, 4, 5]]
    """

    duplicates = [c for c, n in Counter(channel_list).items() if n > 1]
    if len(duplicates):
        warnings.warn("Warning: Channel(s) {} listed more than once".format(duplicates))

    ranges = [[]]
    for channel in channel_list:
        if len(ranges[-1]) and abs(channel - ranges[-1][-1]) != 1:
            ranges.append([])
        ranges[-1].append(channel)
    return ranges


def fmt_channel_ranges(channels, shorten_seq=5, rs="tm", c_sep="_", zp=2):
    """String of channel numbers separated with delimiters with consecutive channels
    are shortened when sequence length above threshold.

    Args:
        channels: list of channels
        shorten_seq: number of consecutive channels to be shortened. (default: 5)
        rs: range delimiter (default: 'tm')
        c_sep: channel delimiter (default: '_')
        zp: zero pad channel numbers (default: 2)

    Returns: String of channels in order they appeared in the list of channels.

    Example: fmt_channel_ranges([[1], [3], [5, 6, 7, 8, 9, 10]]) -> 01_03_05tm10
    """
    c_ranges = channel_ranges(channels)
    range_strings = [c_sep.join(["{c:0{zp}}".format(c=c, zp=zp) for c in c_seq])
                     if len(c_seq) < shorten_seq
                     else "{start:0{zp}}{rs}{end:0{zp}}".format(start=c_seq[0], end=c_seq[-1], rs=rs, zp=zp)
                     for c_seq in c_ranges]
    return c_sep.join(range_strings)


def fmt_time(s, minimal=True):
    """
    Args:
        s: time in seconds (float for fractional)
        minimal: Flag, if true, only return strings for times > 0, leave rest outs
    Returns: String formatted 99h 59min 59.9s, where elements < 1 are left out optionally.

    """
    ms = s - int(s)
    s = int(s)
    if s < 60 and minimal:
        return "{s:02.3f}s".format(s=s + ms)

    m, s = divmod(s, 60)
    if m < 60 and minimal:
        return "{m:02d}min {s:02.3f}s".format(m=m, s=s + ms)

    h, m = divmod(m, 60)
    return "{h:02d}h {m:02d}min {s:02.3f}s".format(h=h, m=m, s=s + ms)


def get_needed_channels(cli_args=None):
    """Gets a list of channels that are needed in order to process a given channel.
    """
    if cli_args is None:
        import argparse
        parser = argparse.ArgumentParser(description=get_needed_channels.__doc__)
        parser.add_argument('probe_file', nargs=1,
                            help="""Phe probe file to be used""")
        parser.add_argument("groups", nargs='+',
                            help="""A list of groups""")
        parser.add_argument("-f", "--filenames", action='store_true',
                            help="""Returns a list of open-ephys continuous filenames, instead of a list of channel
                                    numbers""")
        parser.add_argument("-n", "--node", type=int,
                            help="""A node number for the filenames (default 100)""")
        parser.add_argument("--zerobased", action='store_true',
                            help="Use klusta zero-based convention instead of open-ephys 1-based one")
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

    layout = run_prb(probe_file)

    channels = []
    for g in groups:
        channels.extend(layout['channel_groups'][g]['channels'])
        if 'reference' in layout['channel_groups']:
            channels.extend(layout['channel_groups'][g]['reference'])

    if 'ref_a' in layout:
        channels.extend(layout['ref_a'])

    if 'ref_b' in layout:
        channels.extend(layout['ref_b'])

    if not zero_based:
        channels = [c + 1 for c in channels]
    channels = set(channels)

    if do_filenames:
        fnames = [str(node) + '_CH' + str(c) + '.continuous' for c in channels]
        print('\n'.join(fnames))
    else:
        print(' '.join(map(str, channels)))