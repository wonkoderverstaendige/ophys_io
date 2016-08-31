import re
import sys
import os.path as op
import logging
import warnings
import numpy as np

SIZE_HEADER = 1024  # size of header in B
NUM_SAMPLES = 1024  # number of samples per record
SIZE_RECORD = 2070  # total size of record (2x1024 B samples + record header)
REC_MARKER = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 255], dtype=np.uint8)
NAME_TEMPLATE = '{proc_node:d}_CH{channel:d}.continuous'

# HEADER_REGEX = re.compile("header\.([\d\w.\s]+).=.'*([^;']+)'*")
HEADER_REGEX = re.compile("header\.([\d\w.\s]+).=.\'*([^;\']+)\'*")


def gather_files(input_directory, channels, proc_node, template=NAME_TEMPLATE):
    """Return list of paths to valid input files for the input directory."""
    file_names = [op.join(input_directory, template.format(proc_node=proc_node, channel=chan))
                  for chan in channels]
    is_file = {f: op.isfile(f) for f in file_names}
    try:
        assert all(is_file.values())
    except AssertionError:
        print(IOError("Input files not found: {}".format([f for f, exists in is_file.items() if not exists])))
        sys.exit(1)
    # assert all([f for f in file_names if op.isfile(f)]) == len(file_names)
    logging.debug('File names gathered: {}'.format(file_names))
    return file_names


# data type of .continuous open ephys 0.2x file format header
def header_dt(size_header=SIZE_HEADER):
    return np.dtype([('Header', 'S%d' % size_header)])


# data type of individual records, n Bytes
# (2048 + 22) Byte = 2070 Byte total typically if full 1024 samples
def data_dt(num_samples=NUM_SAMPLES):
    return np.dtype([('timestamp', np.int64),  # 8 Byte
                     ('n_samples', np.uint16),  # 2 Byte
                     ('rec_num', np.uint16),  # 2 Byte
                     ('samples', ('>i2', num_samples)),  # 2 Byte each x 1024 typ.
                     ('rec_mark', (np.uint8, len(REC_MARKER)))])  # 10 Byte


def check_headers(oe_file):
    """Check that length, sampling rate, buffer and block sizes of a list of open-ephys ContinuousFiles are
    identical and return them in that order."""
    # Check oe_file make sense (same size, same sampling rate, etc.
    num_records = [f.num_records for f in oe_file]
    sampling_rates = [f.header['sampleRate'] for f in oe_file]
    buffer_sizes = [f.header['bufferSize'] for f in oe_file]
    block_sizes = [f.header['blockLength'] for f in oe_file]

    assert len(set(num_records)) == 1
    assert len(set(sampling_rates)) == 1
    assert len(set(buffer_sizes)) == 1
    assert len(set(block_sizes)) == 1

    return num_records[0], sampling_rates[0], buffer_sizes[0], block_sizes[0]


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


class ContinuousFile:
    """Single .continuous file. Generates chunks of data."""

    # TODO: Allow record counts and offsets

    def __init__(self, path):
        self.path = op.abspath(op.expanduser(path))
        self.file_size = op.getsize(self.path)
        # Make sure we have full records all the way through
        assert (self.file_size - SIZE_HEADER) % SIZE_RECORD == 0
        self.num_records = (self.file_size - SIZE_HEADER) // SIZE_RECORD
        self.duration = self.num_records

    def __enter__(self):
        self.header = self._read_header()
        self.record_dtype = data_dt(self.header['blockLength'])
        self.__fid = open(self.path, 'rb')
        self.__fid.seek(SIZE_HEADER)
        return self

    def _read_header(self):
        return fmt_header(np.fromfile(self.path, dtype=header_dt(), count=1))

    def read_record(self, count=1):
        buf = np.fromfile(self.__fid, dtype=self.record_dtype, count=count)

        # make sure offsets are likely correct
        assert np.array_equal(buf[0]['rec_mark'], REC_MARKER)
        return buf['samples'].reshape(-1)

    def next(self):
        return self.read_record() if self.__fid.tell() < self.file_size else None

    def __exit__(self, *args):
        self.__fid.close()
