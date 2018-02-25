import os.path as op
from ..lib import tools, Streamer
import numpy as np
import logging
from .open_ephys import NUM_SAMPLES

FMT_NAME = 'DAT'
FMT_FEXT = '.dat'

DEFAULT_DTYPE = 'int16'
DEFAULT_ITEMSIZE = 2
DEFAULT_SAMPLING_RATE = 3e4
AMPLITUDE_SCALE = 1 / 2 ** 10

MAX_NUM_CHANNELS_GUESS = 1024
POSSIBLE_DTYPES = ['float16', 'float32', 'float64', 'int16', 'int32', 'uint16']

logger = logging.getLogger(__name__)


class DataStreamer(Streamer.Streamer):
    def __init__(self, target_path, config, *args, **kwargs):
        super(DataStreamer, self).__init__(*args, **kwargs)
        self.target_path = target_path
        self.dtype = np.dtype('int16')
        logger.debug('DAT-File Streamer Initialized at {}!'.format(target_path))
        self.cfg = config

    def reposition(self, offset):
        logger.debug('Rolling to position {}'.format(offset))

        n_channels = self.buffer.buffer.shape[0]
        byte_offset = offset * n_channels * self.dtype.itemsize * NUM_SAMPLES
        dtype = self.buffer.buffer.dtype

        with open(self.target_path, 'rb') as dat_file:
            dat_file.seek(byte_offset)
            chunk = np.fromfile(dat_file, count=self.buffer.buffer.size, dtype=self.dtype).reshape(-1,
                                                                                                   n_channels).T.astype(
                dtype) * AMPLITUDE_SCALE

        self.buffer.put_data(chunk)


def detect(base_path, pre_walk=None):
    """Checks for existence of a/multiple .dat file(s) at the target path.

    Args:
        base_path: Directory to search in.
        pre_walk: Tuple from previous path_content call (root, dirs, files)

    Returns:
        None if no data set found, else string
    """
    root, dirs, files = tools.path_content(base_path) if pre_walk is None else pre_walk

    logger.debug('Looking for .dat files'.format(dirs, files))
    dat_files = [f for f in files if tools.fext(f) == '.dat']
    logger.debug('{} .dat files found: {}'.format(len(dat_files), dat_files))
    if not len(dat_files):
        return None
    elif len(dat_files) == 1:
        return '{}-File'.format(FMT_NAME)
    else:
        return '{}x {}'.format(len(dat_files), FMT_NAME)


def guess_n_channels(base_path, dtype=DEFAULT_DTYPE, n_channels_max=MAX_NUM_CHANNELS_GUESS, n_bytes=1024):
    """Proper sample alignment most likely has smaller sample diff than when samples of
    channels are mixed up."""

    # FIXME: Return sorted list of candidates
    def mean_abs_diff(arr, n):
        return np.mean(np.abs(np.diff(arr[:arr.shape[0] - arr.shape[0] % (n + 1)].reshape(-1, n + 1), axis=0)))

    chunk = np.fromfile(base_path, dtype=dtype, count=n_bytes * n_channels_max)
    costs = [mean_abs_diff(chunk, n) for n in range(n_channels_max)]
    return int(np.argmin(costs) + 1)


def guess_dtype(base_path, n_bytes=2 ** 16, arr=None):
    # TODO: Finish this up...
    np_dtypes = list(map(np.dtype, POSSIBLE_DTYPES))
    chunk = np.fromfile(base_path, dtype='byte', count=n_bytes)

    ftypes = [dt for dt in POSSIBLE_DTYPES if 'float' in dt]
    fmaxes = [np.finfo(flt).max for flt in ftypes]

    itypes = [dt for dt in POSSIBLE_DTYPES if 'int' in dt]
    imaxes = [np.iinfo(it).max for it in itypes]

    dtypes = ftypes + itypes
    dmaxes = fmaxes + imaxes
    # print(list(zip(dtypes, dmaxes)))

    # # Float -> check for nans or huge exponents
    #
    # # uint or int? -> Check which is larger, int16 or uint16
    #
    # # (u)int16 or (u)int32? -> Non-normalized much larger, and first half < second half!
    #
    # relmus = []
    # for idx, dt in enumerate(POSSIBLE_DTYPES):
    #     relmus.append((dt, np.mean(np.abs(chunk.view(dt))) / dmaxes[idx]))
    #
    # res = []
    # for idx, dt in enumerate(np_dtypes):
    #     mu = np.mean(np.abs(chunk.view(dt)))
    #     res.append(np.Inf if np.isnan(mu) else mu)
    # print('{}\t: mu={:.2f}'.format(dt, res))


def guess_sampling_rate(arr):
    return DEFAULT_SAMPLING_RATE


def metadata(base_path, *args, **kwargs):
    if 'dtype' not in kwargs or kwargs['dtype'] is None:
        dtype = DEFAULT_DTYPE
        guess_dtype(base_path)
        logger.warning('Missing dtype parameter. Defaulting to {} without guessing.'.format(dtype))
    else:
        dtype = kwargs['dtype']

    if 'n_channels' not in kwargs or kwargs['n_channels'] is None:
        logger.warning('Channel number not given. Guessing between 1 and {} channels...'.format(MAX_NUM_CHANNELS_GUESS))
        n_channels = guess_n_channels(base_path, dtype=dtype, n_channels_max=MAX_NUM_CHANNELS_GUESS)
        logger.warning('{} seems to have {} channels'.format(base_path, n_channels))
    else:
        n_channels = kwargs['n_channels']

    if 'sampling_rate' not in kwargs:
        sampling_rate = DEFAULT_SAMPLING_RATE
        logger.warning(
            'Missing sampling rate. Defaulting to {} kHz without guessing.'.format(DEFAULT_SAMPLING_RATE / 1e3))
    else:
        sampling_rate = kwargs['sampling_rate']

    return {'HEADER': {'sampling_rate': sampling_rate,
                       'block_size': 1,
                       'n_samples': op.getsize(base_path) / DEFAULT_ITEMSIZE / n_channels},
            'DTYPE': dtype,
            'CHANNELS': {'n_channels': n_channels},
            'INFO': None,
            'SIGNALCHAIN': None,
            'FPGA_NODE': None,
            'AUDIO': None}


def fill_buffer(target, buffer, offset, **kwargs):
    # channels = kwargs['channels']
    n_channels = buffer.shape[0]
    byte_offset = offset * n_channels * DEFAULT_ITEMSIZE * NUM_SAMPLES
    n_samples = n_channels * buffer.shape[1]
    dtype = kwargs['dtype'] if 'dtype' in kwargs else 'int16'

    with open(target, 'rb') as dat_file:
        dat_file.seek(byte_offset)
        logger.debug('offset: {}, byte_offset: {}, count: {}'.format(offset, byte_offset, buffer.shape[1]))
        chunk = np.fromfile(dat_file, count=n_samples, dtype=dtype).reshape(-1, n_channels).T.astype(
            'float32') * AMPLITUDE_SCALE
        buffer[:] = chunk
