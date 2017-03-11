from six import exec_
import os.path as op
import warnings
from collections import Counter


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
        path: Path to probe file.

    Returns:
        Tuple of lists of channels.
    """
    channels = sum([prb['channel_groups'][cg]['channels'] for cg in sorted(prb['channel_groups'])], [])
    dead_channels = prb['dead_channels']

    return channels, dead_channels

def make_prb(path, stuff):
    raise NotImplementedError


def make_prm(path, stuff):
    raise NotImplementedError


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
    ms = s-int(s)
    s = int(s)
    if s < 60 and minimal:
        return "{s:02.3f}s".format(s=s+ms)

    m, s = divmod(s, 60)
    if m < 60 and minimal:
        return "{m:02d}min {s:02.3f}s".format(m=m, s=s+ms)

    h, m = divmod(m, 60)
    return "{h:02d}h {m:02d}min {s:02.3f}s".format(h=h, m=m, s=s+ms)
