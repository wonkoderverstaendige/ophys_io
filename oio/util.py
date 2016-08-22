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


def fmt_channel_ranges(channels, shorten_seq=5, rs="tm", cs="_", zp=2):
    """String of channel numbers separated with delimiters with consecutive channels
    are shortened when sequence length above threshold.

    Args:
        channels: list of channels
        shorten_seq: number of consecutive channels to be shortened. (default: 5)
        rs: range delimiter (default: 'tm')
        cs: channel delimiter (default: '_')
        zp: zero pad channel numbers (default: 2)

    Returns: String of channels in order they appeared in the list of channels.

    Example: fmt_channel_ranges([[1], [3], [5, 6, 7, 8, 9, 10]]) -> 01_03_05tm10
    """
    cr = channel_ranges(channels)
    range_strings = [cs.join(["{c:0{zp}}".format(c=c, zp=zp) for c in cr])
                     if len(cr) < shorten_seq
                     else "{start:0{zp}}{rs}{end:0{zp}".format(start=cr[0], end=cr[-1], rs=rs, zp=zp)
                     for cr in cr]
    return cs.join(range_strings)