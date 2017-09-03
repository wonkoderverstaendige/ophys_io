# -*- coding: utf-8 -*-

import re
import os
import sys
import xml.etree.ElementTree as ETree
from ..lib import tools, Streamer
import numpy as np
import logging

FMT_NAME = 'OE'
FMT_FEXT = '.continuous'

SIZE_HEADER = 1024  # size of header in B
NUM_SAMPLES = 1024  # number of samples per record
SIZE_RECORD = 2070  # total size of record (2x1024 B samples + record header)
REC_MARKER = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 255], dtype=np.uint8)
# NAME_TEMPLATE = '{proc_node:}_CH{channel:d}.continuous'
NAME_TEMPLATE = '{proc_node:}_CH{channel:d}_{sub_id:d}.continuous'


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
                    ('rec_mark', (np.uint8, 10))])  # 10 Byte


class DataStreamer(Streamer.Streamer):
    def __init__(self, target_path, config, *args, **kwargs):
        super(DataStreamer, self).__init__(*args, **kwargs)
        self.target_path = target_path
        logger.debug('Open Ephys Streamer Initialized at {}!'.format(target_path))
        self.cfg = config

    def reposition(self, offset):
        logger.debug('Rolling to position {}'.format(offset))
        dtype = np.dtype(self.buffer.np_type)
        n_samples = self.buffer.buffer.shape[1]

        channels = range(self.buffer.n_channels)
        self.files = [
            (channel, os.path.join(self.target_path, NAME_TEMPLATE.format(self.cfg['FPGA_NODE'], channel + 1)))
            for
            channel in channels]

        for sf in self.files:
            data = read_record(sf[1], offset=offset)[:n_samples]
            self.buffer.put_data(data, channel=sf[0])

def gather_files(input_directory, channels, proc_node, sub_id=0, template=NAME_TEMPLATE):
    """Return list of paths to valid input files for the input directory for the given sub-id."""
    file_names = [os.path.join(input_directory, template.format(proc_node=proc_node, channel=channel, sub_id=sub_id))
                  for channel in channels]
    is_file = {f: os.path.isfile(f) for f in file_names}
    if all(is_file.values()):
        return file_names
    else:
        raise(IOError("Input files not found: {}".format([f for f, exists in is_file.items() if not exists])))

def check_headers(files):
    """Check that length, sampling rate, buffer and block sizes of a list of open-ephys ContinuousFiles are
    identical and return them in that order."""
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


def fill_buffer(target, buffer, offset, *args, **kwargs):
    count = buffer.shape[1] // NUM_SAMPLES + 1
    logger.debug('count: {}, buffer: {} '.format(count, buffer.shape))
    channels = kwargs['channels']
    node_id = kwargs['node_id']
    for c in channels:
        buffer[c, :] = \
            read_record(os.path.join(target, NAME_TEMPLATE.format(
                node_id=node_id,
                channel=c + 1)),
                        count=count,
                        offset=offset)[:buffer.shape[1]]

def read_header(filename):
    """Return dict with .continuous file header content."""
    # TODO: Compare headers, should be identical except for channel

    # 1 kiB header string data type
    header = read_segment(filename, offset=0, count=1, dtype=HEADER_DT)
    return fmt_header(header)


def fmt_header(header_str):
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
    """Read segment of a file from [offset] for [count]x[dtype]"""
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
    # if not os.path.isdir(base_path) and os.path.splitext(base_path)[1]!='.continuous':
    #     logger.warning('Path {} neither a directory nor a .continuous file.'.format(base_path))
    #     return None
    root, dirs, files = tools.path_content(base_path) if pre_walk is None else pre_walk

    # FIXME: Do once for all files. If single file, indicate
    for f in files:
        if tools.fext(f) in ['.continuous']:
            fv = config_xml(base_path)['INFO']['VERSION']
            return "{}_v{}".format(FMT_NAME, fv if fv else '???')
    else:
        return None


def find_settings_xml(base_dir):
    """Search for the settings.xml file in the base directory.

    :param base_dir: Base directory of data set

    :return: Path to settings.xml relative to base_dir
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
        string of NodeID (e.g. '106')
    """
    # chain = config_xml(base_dir)['SIGNALCHAIN']
    nodes = [p['attrib']['NodeId'] for p in chain_dict if p['type'] == 'PROCESSOR' and 'FPGA' in p['attrib']['name']]
    logger.info('Found FPGA node(s): {}'.format(nodes))
    if len(nodes) == 1:
        return nodes[0]
    if len(nodes) > 1:
        raise BaseException('Multiple FPGA nodes found. (Good on you!) {}'.format(nodes))
    else:
        raise BaseException('Node ID not found in xml dict {}'.format(chain_dict))


def config(base_dir, *args, **kwargs):
    """Get recording/data set configuration from open ephys GUI settings.xml file and the header
    of files from the data set

    Args:
        base_dir: path to the data set

    Returns:
        Dictionary with configuration entries. (INFO, SIGNALCHAIN, HEADER, AUDIO, FPGA_NODE)"""
    cfg = config_xml(base_dir)
    cfg['FPGA_NODE'] = _fpga_node(cfg['SIGNALCHAIN'])
    cfg['HEADER'] = config_header(base_dir, cfg['FPGA_NODE'])
    cfg['DTYPE'] = DEFAULT_DTYPE

    if 'n_channels' not in kwargs or kwargs['n_channels'] is None:
        logger.warning('Channel number not given. Defaulting to 64.'.format(base_dir))
        n_channels = guess_n_channels(base_dir, cfg['FPGA_NODE'])
        # logger.warning('{} seems to have {} channels'.format(base_dir, n_channels))
    else:
        n_channels = kwargs['n_channels']

    cfg['CHANNELS'] = {'n_channels': n_channels}
    return cfg


def config_header(base_dir, fpga_node='106'):
    """Reads header information from open ephys .continuous files in target path.
    This returns some "reliable" information about the sampling rate."""
    # Data file header (reliable sampling rate information)
    # FIXME: Make sure all headers agree...
    file_name = os.path.join(base_dir, '{}_CH1_0.continuous'.format(fpga_node))
    header = read_header(file_name)
    fs = header['sampleRate']
    n_samples = int(os.path.getsize(file_name) - SIZE_HEADER)
    n_blocks = n_samples / SIZE_RECORD
    assert n_samples % SIZE_RECORD == 0

    logger.info('Fs = {:.2f}Hz, {} blocks, {} samples, {}'
                .format(fs, n_blocks, n_samples, tools.fmt_time(n_samples / fs)))

    return dict(n_blocks=int(n_blocks),
                block_size=NUM_SAMPLES,
                n_samples=int(n_samples),
                sampling_rate=fs)


def config_xml(base_dir):
    """Reads Open Ephys XML settings file and returns dictionary with relevant information.
        - Info field
            Dict(GUI version, date, OS, machine),
        - Signal chain
            List(Processor or Switch dict(name, nodeID). Signal chain is returned in ORDER OF OCCURRENCE
            in the xml file, this represents the order in the signal chain. The nodeID does NOT reflect this order, but
            but the order of assembling the chain.
        - Audio
            Int bufferSize
        - Header
            Dict(data from a single file header, i.e. sampling rate, blocks, etc.)

    Args:
        base_dir: Path to settings.xml file

    Returns:
        Dict{INFO, SIGNALCHAIN, AUDIO, HEADER}
    """

    # Settings.xml file
    xml_path = find_settings_xml(base_dir)
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

    return dict(INFO=info,
                SIGNALCHAIN=chain,
                AUDIO=audio)


def guess_n_channels(base_path, fpga_node, *args, **kwargs):
    # FIXME: Hardcoding. Poor man's features.
    return 64


# # data type of individual records, n Bytes
# # (2048 + 22) Byte = 2070 Byte total typically if full 1024 samples
# def data_dt(num_samples=NUM_SAMPLES):
#     return np.dtype([('timestamp', np.int64),  # 8 Byte
#                      ('n_samples', np.uint16),  # 2 Byte
#                      ('rec_num', np.uint16),  # 2 Byte
#                      ('samples', ('>i2', num_samples)),  # 2 Byte each x 1024 typ.
#                      ('rec_mark', (np.uint8, len(REC_MARKER)))])  # 10 Byte
# class ContinuousFile:
#     """Single .continuous file. Generates chunks of data."""
#
#     # TODO: Allow record counts and offsets
#
#     def __init__(self, path):
#         self.path = op.abspath(op.expanduser(path))
#         self.file_size = op.getsize(self.path)
#         # Make sure we have full records all the way through
#         assert (self.file_size - SIZE_HEADER) % SIZE_RECORD == 0
#         self.num_records = (self.file_size - SIZE_HEADER) // SIZE_RECORD
#         self.duration = self.num_records
#
#     def __enter__(self):
#         self.header = self._read_header()
#         self.record_dtype = data_dt(self.header['blockLength'])
#         self.__fid = open(self.path, 'rb')
#         self.__fid.seek(SIZE_HEADER)
#         return self
#
#     def _read_header(self):
#         return fmt_header(np.fromfile(self.path, dtype=header_dt(), count=1))
#
#     def read_record(self, count=1):
#         buf = np.fromfile(self.__fid, dtype=self.record_dtype, count=count)
#
#         # make sure offsets are likely correct
#         assert np.array_equal(buf[0]['rec_mark'], REC_MARKER)
#         return buf['samples'].reshape(-1)
#
#     def next(self):
#         return self.read_record() if self.__fid.tell() < self.file_size else None
#
#     def __exit__(self, *args):
#         self.__fid.close()

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
        return fmt_header(np.fromfile(self.path, dtype=HEADER_DT, count=1))

    def read_record(self, count=1):
        buf = np.fromfile(self.__fid, dtype=self.record_dtype, count=count)

        # make sure offsets are likely correct
        assert np.array_equal(buf[0]['rec_mark'], REC_MARKER)
        return buf['samples'].reshape(-1)

    def next(self):
        return self.read_record() if self.__fid.tell() < self.file_size else None

    def __exit__(self, *args):
        self.__fid.close()
