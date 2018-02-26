# -*- coding: utf-8 -*-

import re
import os
import xml.etree.ElementTree as ETree
from ..lib import tools, Streamer
import numpy as np
import logging
from pathlib import Path
from pprint import pformat

LOG_LEVEL_VERBOSE = 5

FMT_NAME = 'OE'
FMT_FEXT = '.continuous'

SIZE_HEADER = 1024  # size of header in B
NUM_SAMPLES = 1024  # number of samples per record
SIZE_RECORD = 2070  # total size of record (2x1024 B samples + record header)
SIZE_DATA = 2 * NUM_SAMPLES
REC_MARKER = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 255], dtype=np.uint8)

NAME_TEMPLATE = '{proc_node}_{channel_type}{channel}{sub_id}.continuous'  # sub id: [_0, _1], or none
# NAME_TEMPLATE_NO_SUB_ID = '{proc_node:}_CH{channel:d}.continuous'


AMPLITUDE_SCALE = 1 / 2 ** 10

# data type of .continuous open ephys 0.2x file format header
# HEADER_REGEX = re.compile("header\.([\d\w.\s]+).=.\'*([^;\']+)\'*")
HEADER_REGEX = re.compile("header\.([\d\w\.\s]{1,}).=.\'*([^\;\']{1,})\'*")
HEADER_DT = np.dtype([('Header', 'S%d' % SIZE_HEADER)])
DEFAULT_DTYPE = 'int16'

logger = logging.getLogger(__name__)

# (2048 + 22) Byte = 2070 Byte total
# FIXME: The rec_mark comes after the samples. Currently data is read assuming full NUM_SAMPLE record!
DATA_DT = np.dtype([('timestamp', np.int64),  # 8 Byte
                    ('n_samples', np.uint16),  # 2 Byte
                    ('rec_num', np.uint16),  # 2 Byte
                    ('samples', ('>i2', NUM_SAMPLES)),  # 2 Byte each x 1024 typ.
                    ('rec_mark', (np.uint8, len(REC_MARKER)))])  # 10 Byte


class ContinuousFile:
    """Single .continuous file. Generates chunks of data."""

    # TODO: Allow record counts and offsets

    def __init__(self, path):
        self.path = os.path.abspath(os.path.expanduser(path))
        self.file_size = os.path.getsize(self.path)
        # Make sure we have full records all the way through
        assert (self.file_size - SIZE_HEADER) % SIZE_RECORD == 0
        self.num_records = (self.file_size - SIZE_HEADER) // SIZE_RECORD
        self.duration = self.num_records

    def __enter__(self):
        self.header = self._read_header()
        self.record_dtype = DATA_DT  # data_dt(self.header['blockLength'])
        self.__fid = open(self.path, 'rb')
        self.__fid.seek(SIZE_HEADER)
        return self

    def _read_header(self):
        return format_header(np.fromfile(self.path, dtype=HEADER_DT, count=1))

    def read_record(self, count=1):
        buf = np.fromfile(self.__fid, dtype=self.record_dtype, count=count)

        # make sure offsets are likely correct
        assert np.array_equal(buf[0]['rec_mark'], REC_MARKER)
        return buf['samples'].reshape(-1)

    def next(self):
        return self.read_record() if self.__fid.tell() < self.file_size else None

    def __exit__(self, *args):
        self.__fid.close()


class DataStreamer(Streamer.Streamer):
    def __init__(self, target_path, config, *args, **kwargs):
        super(DataStreamer, self).__init__(*args, **kwargs)
        self.target_path = target_path
        logger.debug('Open Ephys Streamer Initialized at {}!'.format(target_path))
        self.cfg = config
        self.files = None

    def reposition(self, offset):
        logger.debug('Rolling to position {}'.format(offset))
        # dtype = np.dtype(self.buffer.np_type)
        n_samples = self.buffer.buffer.shape[1]

        channels = range(self.buffer.n_channels)
        self.files = [
            (channel, os.path.join(self.target_path, NAME_TEMPLATE.format(self.cfg['FPGA_NODE'], channel + 1)))
            for
            channel in channels]

        for sf in self.files:
            data = read_record(sf[1], offset=offset)[:n_samples]
            self.buffer.put_data(data, channel=sf[0])


def _ids_from_fname(fn, channel_type='CH'):
    """Extract channel number and sub_id from a .continuous file of expected form
    {NODE_ID}_CH{CHANNEL}[_{SUB_ID}].continuous. The underscore is used as indicator of the file name format.
    Args:
        fn: Filename

    Returns:
        Channel number, sub_id
    """
    id_str = fn[fn.index(channel_type) + 2:fn.index('.continuous')]
    if '_' in id_str:
        # has sub_id
        sub_id = id_str[id_str.index('_') + 1:]
        channel = id_str[:id_str.index('_')]
    else:
        # no sub_id
        sub_id = -1
        channel = id_str

    try:
        sub_id = int(sub_id)
        channel = int(channel)
    except ValueError:
        raise ValueError("Couldn't parse file name {}, not matching expected pattern.".format(fn))

    return sub_id, channel


def gather_files(target_directory, proc_node, channels='*', channel_type='CH',
                 sub_id=-1, scan_sub_ids=True, template=NAME_TEMPLATE):
    """Return dictionary of list of paths to valid input files for the input directory, keyed by subset ids.
    Sub_id is the numerical suffix after channel number in file name, indicating additional recordings
    subsets in the same data folder. Also checks that all sub_ids have the same number of files.
    Because that would cause headaches otherwise...

    The annoyance with gathering files here is that we don't know the number of channels or the number of sub_ids.
    There is no/I don't know a wildcard pattern to catch all channel numbers AND separate the sub_ids, as channel
    numbers aren't padded. Looks like we'll need to do this "manually". Furthermore, the sub_id may start at _0
    without non-sufficed companions.

    What we do is to gather up all channel_type matching .continuous files, find the set of sub_ids and channel
    numbers, then loop over the combination of both and rebuilding those file names. This is kind of backwards,
    but doesn't require too much prior knowledge of the search space.

    E.g. file names: [100_CH34.continuous, 100_CH34_0.continuous], 100_CH1.continuous, 106_CH1024_59.continuous

    Args:
        target_directory: path to directory to be scanned
        proc_node: Processing node origin of files, typically 100
        channels: Channel number or glob pattern for multiple channels, default all: *
        channel_type: Type of channel to look for (CH, ADC, AUX). Only one can be used right now.

        sub_id: Starting sub_id. If sub_id is None, no suffix will be appended. Else, numerical.
        scan_sub_ids: Only scan for a single sub_id, or scan through range incrementally until no files found.
        template: Glob pattern template for proc_node, channels, sub_id to match .continuous file names.

    Returns: dict {sub_id: {channel: {proc_node: proc_node, channel_type: channel_type, filename:filename}}

    TODO: Channel selection list
    TODO: Multiple channel types
    """
    logger.debug('Gathering .continuous files with template {}'.format(template))
    logger.debug('Channel type: {}, starting sub_id: {}, channel range: {}, scanning subsets: {}'.format(
        channel_type, sub_id, channels, scan_sub_ids))

    target_path = Path(target_directory).resolve()
    files = {}

    # This glob catches ALL channels/sub_ids present in the target directories for the proc_node!
    glob = template.format(proc_node=proc_node, channel_type=channel_type, channel='*', sub_id='')
    globbed = [g.name for g in target_path.glob(glob)]

    globbed_sub_ids, globbed_channels = map(tuple, map(set, zip(*[_ids_from_fname(f, channel_type) for f in globbed])))
    logger.debug('Found sub ids: {}, channels: {}'.format(globbed_sub_ids, globbed_channels))

    globbed_channels = [g - 1 for g in globbed_channels]
    # override sub_id search space to singular instance if we aren't scanning them all
    if not scan_sub_ids:
        if sub_id not in globbed_sub_ids:
            raise ValueError('Requested sub_id {} not found in files at target.'.format(sub_id))
        globbed_sub_ids = [sub_id]

    # Build the return dictionary by building file names from the template and check their existence
    for sid in sorted(globbed_sub_ids):
        sub_files = {}
        for channel in sorted(globbed_channels):
            filename = template.format(proc_node=proc_node, channel=channel + 1, channel_type=channel_type,
                                       sub_id='' if sid is -1 else '_{}'.format(sid))
            file_path = target_path / filename
            # Check file existence
            if not file_path.exists():
                raise FileNotFoundError('Sub_ids and channels not matching up at {}'.format(file_path))

            sub_files[channel] = {'PROC_NODE': proc_node, 'CHANNEL_TYPE': channel_type, 'CHANNEL': channel,
                                      'FILEPATH': str(file_path), 'FILENAME': filename}
        files[sid] = {'FILES': sub_files}
    return files


def check_continuous_headers(files):
    """Check that length, sampling rate, buffer and block sizes of a list of open-ephys ContinuousFiles are
    identical and return them in that order."""
    # FIXME: Not clear enough this reads only from ContinuousFiles, not raw file paths
    # Check oe_file make sense (same size, same sampling rate, etc.
    num_records = [f.num_records for f in files]
    sampling_rates = [f.header['sampleRate'] for f in files]
    buffer_sizes = [f.header['bufferSize'] for f in files]
    block_sizes = [f.header['blockLength'] for f in files]

    assert len(set(num_records)) == 1
    assert len(set(sampling_rates)) == 1
    assert len(set(buffer_sizes)) == 1
    assert len(set(block_sizes)) == 1

    return num_records[0], sampling_rates[0], buffer_sizes[0], block_sizes[0]


# def fill_buffer(target, buffer, offset, *args, **kwargs):
#     # count = buffer.shape[1] // NUM_SAMPLES + 1
#     # logger.debug('count: {}, buffer: {} '.format(count, buffer.shape))
#     # channels = kwargs['channels']
#     # node_id = kwargs['node_id']
#     raise NotImplemented
#     # CURRENT ISSUE: SHOULD USE GATHER_FILES, NOT IT'S OWN IDEA OF HOW TO FIND FILES!
#     # for c in channels:
#     #     buffer[c, :] = \
#     #         read_record(os.path.join(target, NAME_TEMPLATE.format(
#     #             node_id=node_id,
#     #             channel=c + 1)),
#     #                     count=count,
#     #                     offset=offset)[:buffer.shape[1]]


def read_header(filename):
    """Return dict with .continuous file header content."""
    # TODO: Compare headers, should be identical except for channel

    # 1 kiB header string data type
    header = read_segment(filename, offset=0, count=1, dtype=HEADER_DT)
    return format_header(header)


def format_header(header_str):
    """Extract fields from header string and convert into types as needed.

    Args:
        header_str: String from .continuous file header (first 1024 bytes)

    Returns:
        header_dict: Dictionary of fields with proper data type.
    """
    # Stand back! I know regex!
    # Annoyingly, there is a newline character missing in the header_str (version/header_bytes)
    header_str = str(header_str[0][0]).rstrip(' ')
    header_dict = {group[0]: group[1] for group in HEADER_REGEX.findall(header_str)}
    for key in ['bitVolts', 'sampleRate']:
        header_dict[key] = float(header_dict[key])
    for key in ['blockLength', 'bufferSize', 'header_bytes', 'channel']:
        header_dict[key] = int(header_dict[key] if not key == 'channel' else header_dict[key][2:])
    return header_dict


def read_segment(filename, offset, count, dtype):
    """Read segment of a file from [offset] for [count]x[dtype] bytes"""
    with open(filename, 'rb') as fid:
        fid.seek(int(offset))
        segment = np.fromfile(fid, dtype=dtype, count=count)
    return segment


def read_record(filename, offset=0, count=30, dtype=DATA_DT):
    # FIXME: Stupid undocumented magic division of return value...
    return read_segment(filename, offset=SIZE_HEADER + offset * SIZE_RECORD, count=count, dtype=dtype)['samples'] \
               .ravel() \
               .astype(np.float32) * AMPLITUDE_SCALE


def detect(base_path, pre_walk=None):
    """Checks for existence of an open ephys formatted data set in the root directory.

    Args:
        base_path: Path to search at. For OE data sets, typically a directory containing
                   .continuous, .events and a settings.xml file.
        pre_walk: Tuple from previous path_content call

    Returns:
        None if no data set found, else a dict of configuration data from settings.xml
    """
    logger.debug('Looking for .continuous data set in {}'.format(base_path))
    root, dirs, files = tools.path_content(base_path) if pre_walk is None else pre_walk

    # FIXME: Do once for all files. If single file, indicate
    # TODO: metadata_from_xml is called twice. Once on detection, once on gathering metadata
    for f in files:
        if tools.fext(f) in ['.continuous']:
            fv = metadata_from_xml(base_path)['INFO']['VERSION']
            return "{}_v{}".format(FMT_NAME, fv if fv else '???')
    else:
        return None


def find_settings_xml(base_dir):
    """Search for the settings.xml file in the base directory.

    Args:
        base_dir: Base directory of data set

    Returns:
        Path to settings.xml relative to base_dir
    """
    _, dirs, files = tools.path_content(base_dir)
    if "settings.xml" in files:
        return os.path.join(base_dir, 'settings.xml')
    else:
        return None


def _fpga_node(chain_dict):
    """Find the FPGA node in the signal chain. Assuming this one was used for recording, will help
    finding the proper .continuous files.

    Args:
        chain_dict: Root directory of data set.

    Returns:
        string of NodeID (e.g. '100')
    """
    nodes = [p['attrib']['NodeId'] for p in chain_dict if p['type'] == 'PROCESSOR' and 'FPGA' in p['attrib']['name']]
    logger.debug('Found FPGA node(s): {}'.format(nodes))
    if len(nodes) == 1:
        return nodes[0]
    if len(nodes) > 1:
        raise BaseException('Multiple FPGA nodes found. (Good on you though!) {}'.format(nodes))
    else:
        raise BaseException('Node ID not found in xml dict {}'.format(chain_dict))


def metadata_from_target(target_dir, channel_type='CH'):
    """Get metadata from directory containing .continuous files. Directories may contain multiple "subsets"
    of recordings that may have been acquired at different points in time. Stupid.

    Args:
        target_dir: path to the data set
        channel_type: AUX, CH or ADC. Default: 'CH'

    Returns:
        Dictionary with configuration entries. (DTYPE, INFO, SIGNALCHAIN, SUBSETS, AUDIO, FPGA_NODE)
    """
    metadata = {'DTYPE': DEFAULT_DTYPE,
                'TARGET': str(Path(target_dir).resolve())}

    # Grab metadata from the settings.xml file in the base directory
    metadata.update(metadata_from_xml(target_dir))

    logger.debug('Searching FPGA node in signal chain')
    metadata['FPGA_NODE'] = _fpga_node(metadata['SIGNALCHAIN'])

    metadata['CHANNEL_TYPE'] = channel_type

    metadata['SUBSETS'] = gather_files(target_dir, metadata['FPGA_NODE'], channel_type=channel_type)

    for sub_id in metadata['SUBSETS']:
        for channel, channel_metadata in metadata['SUBSETS'][sub_id]['FILES'].items():
            channel_metadata.update(metadata_from_file(channel_metadata['FILEPATH']))
        metadata['SUBSETS'][sub_id]['JOINT_HEADERS'] = reduce_files_metadata(metadata['SUBSETS'][sub_id]['FILES'])

    logger.log(level=LOG_LEVEL_VERBOSE, msg=pformat(metadata, indent=2))

    # Channels should be the same for all subsets
    channels = sorted(set([ch for sub_id in metadata['SUBSETS']
                           for ch in metadata['SUBSETS'][sub_id]['JOINT_HEADERS']['CHANNEL']]))
    metadata['CHANNELS'] = channels

    return metadata


def metadata_from_file(file):
    """Read metadata of single .continuous file header/file stats. Checks if the file contains
    complete records based on the size of the file as header size + integer multiples of records.

    Args:
        file: Path to .continuous file

    Returns:
        Dictionary with n_blocks, block_size, n_samples, sampling_rate fields.
    """
    header = read_header(file)
    fs = header['sampleRate']
    n_record_bytes = int(os.path.getsize(file) - SIZE_HEADER)
    n_blocks = n_record_bytes / SIZE_RECORD
    if n_record_bytes % SIZE_RECORD != 0:
        raise ValueError('File {} contains incomplete records!'.format(file))

    n_samples = n_record_bytes - n_blocks * (SIZE_RECORD - SIZE_DATA)
    logger.log(level=LOG_LEVEL_VERBOSE, msg='{}, Fs = {:.2f}Hz, {} blocks, {} samples, {}'
               .format(file, fs, n_blocks, n_samples, tools.fmt_time(n_samples / fs)))

    return dict(n_blocks=int(n_blocks),
                block_size=NUM_SAMPLES,
                n_samples=int(n_samples),
                sampling_rate=fs)


def reduce_files_metadata(files_metadata):
    """Check that length, sampling rate, buffer and block sizes of given files are consistent.

    Args:
        files_metadata: Dictionary of channel metadata dictionaries extracted from file headers/file stats

    Returns:
        Dict of union of metadata sets when all metadata are the same for all items.
    """
    joint_headers = {}
    singular_items = ['n_samples', 'block_size', 'sampling_rate', 'n_blocks']

    files = list(files_metadata.values())
    keys = tuple(set([i for f in files for i in f.keys()]))
    for k in keys:
        property_set = sorted(list(set([f[k] for f in files])))

        # Check that properties that should be the same for all files in a subset are equal
        if k in singular_items and len(property_set) > 1:
            raise ValueError('Found non-singular metadata in {}'.format(k))

        # Singular properties should be reduced to scalar
        property_set = property_set if len(property_set) > 1 else property_set[0]
        joint_headers[k] = property_set

    return joint_headers


def metadata_from_xml(base_dir):
    """Reads Open Ephys settings.xml file and returns dictionary with relevant information.
        - Info field
            Dict(GUI version, date, OS, machine),
        - Signal chain
            List(Processor or Switch dict(name, nodeID). Signal chain is returned in ORDER OF OCCURRENCE
            in the xml file, this represents the order in the signal chain. The nodeID does NOT reflect this order, but
            but the order of assembling the chain.
        - Audio
            Int bufferSize
        - Header (NOT from XML, but from data files)
            Dict(data from a single file header, i.e. sampling rate, blocks, etc.)

    Args:
        base_dir: Path to settings.xml file

    Returns:
        Dict{INFO, SIGNALCHAIN, AUDIO, HEADER}
    """
    logger.debug('Looking for settings.xml in {} ...'.format(base_dir))
    xml_path = find_settings_xml(base_dir)
    logger.debug('Found. Reading in xml metadata...')
    root = ETree.parse(xml_path).getroot()

    # Recording system information
    info = dict(
        VERSION=root.find('INFO/VERSION').text,
        DATE=root.find('INFO/DATE').text,
        OS=root.find('INFO/OS').text,
        MACHINE=root.find('INFO/VERSION').text
    )

    # Signal chain/processing nodes
    sc = root.find('SIGNALCHAIN')
    chain = [dict(type=e.tag, attrib=e.attrib) for e in sc.getchildren()]

    # Audio settings
    audio = root.find('AUDIO').attrib

    md = dict(INFO=info, SIGNALCHAIN=chain, AUDIO=audio)
    logger.log(level=LOG_LEVEL_VERBOSE, msg='Got: {}'.format(pformat(md, indent=2)))
    return md


def guess_n_channels(base_path, fpga_node, *args, **kwargs):
    """What was I thinking?"""
    # FIXME: Hardcoding. Poor man's features.
    raise NotImplemented
    # return 64
