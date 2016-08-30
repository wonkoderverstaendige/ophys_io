import re
import os.path as op
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
    assert len([f for f in file_names if op.isfile(f)]) == len(file_names)
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
        assert (self.file_size-SIZE_HEADER) % SIZE_RECORD == 0
        self.num_records = (self.file_size-SIZE_HEADER) // SIZE_RECORD
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


# def read_header(filename):
#     """Return dict with the content of the 1kiB .continuous file header."""
#     return fmt_header(read_segment(filename, offset=0, count=1, dtype=header_dt()))
#
#
# def read_segment(filename, offset, count, dtype):
#     """Read segment of a file from [offset] for [count]x[dtype]"""
#     with open(filename, 'rb') as fid:
#         fid.seek(offset)
#         # print(count)
#         segment = np.fromfile(fid, dtype=dtype, count=count)
#     return segment
#
#
# def read_records(filename, record_offset=0, record_count=10000,
#                  size_header=SIZE_HEADER, num_samples=NUM_SAMPLES, size_record=SIZE_RECORD):
#     return read_segment(filename, offset=size_header + record_offset * size_record, count=record_count,
#                         dtype=data_dt())
#
#
# def suggest_proc_node(path):
#     """Suggest proc node based on files present in the target directory"""
#     # FIXME: do the actual checking
#     warnings.warn("Proc node suggestion with magic number (100)")
#     return 100


# def make_buffer(n_channels, chunk_size, dtype=np.int16):
#     return np.zeros((n_channels, chunk_size*SIZE_RECORD), dtype=dtype)
#
#
# def data_to_buffer(file_paths, count=1000, buf=None, size_record=SIZE_RECORD):
#     """Read [count] records from [proc_node] file at [filepath] into a buffer."""
#     buf = buf if buf is not None else np.zeros((len(file_paths), count * size_record), dtype='>i2')
#     return buf
#
#
# class ContinuousReader(object):
#     def __init__(self, file_path, *args, **kwargs):
#         self.file_path = os.path.abspath(file_path)
#         assert os.path.isfile(self.file_path)
#         self.file_size = os.path.getsize(self.file_path)
#
#         self.chunk_size = None
#         self.bytes_read = None
#         self.file_size = None
#
#         self.buffer = None
#         self._header = None
#
#         self.header_dt = oe.header_dt()
#         self.data_dt = oe.data_dt()
#
#     @property
#     def header(self):
#         if self._header is None:
#             self._header = oe.read_header(self.file_path)
#         return self._header
#
#     @header.setter
#     def header(self, value):
#         pass
#
#     def read_header(self):
#         """Return dict with .continuous file header content."""
#         # TODO: Compare headers, should be identical except for channel
#
#         # 1 kiB header string data type
#         header = read_segment(self.file_path, offset=0, count=1, dtype=oe.header_dt())
#
#         # Stand back! I know regex!
#         # Annoyingly, there is a newline character missing in the header (version/header_bytes)
#         regex = "header\.([\d\w\.\s]{1,}).=.\'*([^\;\']{1,})\'*"
#         header_str = str(header[0][0]).rstrip(' ')
#         header_dict = {group[0]: group[1] for group in re.compile(regex).findall(header_str)}
#         for key in ['bitVolts', 'sampleRate']:
#             header_dict[key] = float(header_dict[key])
#         for key in ['blockLength', 'bufferSize', 'header_bytes', 'channel']:
#             header_dict[key] = int(header_dict[key] if not key == 'channel' else header_dict[key][2:])
#
#         return header_dict
#
#     def read_chunk(self):
#         pass
#
#     def check_chunk(self):
#         pass
#
#     def check_data(self):
#         pass
#
#     def read_segment(self, offset, count, dtype):
#         """Read segment of a file from [offset] for [count]x[dtype]"""
#         with open(self.file_path, 'rb') as fid:
#             fid.seek(offset)
#             segment = np.fromfile(fid, dtype=dtype, count=count)
#         return segment


# class SliceReader(object):
#     """ Slice reader represents a single slice through all files in a single directory.
#     """
#
#     def __init__(self, *args, **kwargs):
#         self.input_directory = None
#         assert os.path.isdir(os.path.abspath(self.input_directory))
#         self.buffer = None
#         self.headers = None
#         self.files = None
#         self.channels = None
#         self.proc_node = None
#
#     def stack_files(self):
#         pass
#
#     def gather_files(self):
#         base_name = os.path.join(self.input_directory, '{proc_node:d}_CH{channel:d}.continuous')
#         file_names = [base_name.format(proc_node=self.proc_node, channel=chan) for chan in self.channels]
#         for f in file_names:
#             assert os.path.isfile(f)
#
#         # all input files in a single directory should have equal length
#         file_sizes = [os.path.getsize(fname) for fname in file_names]
#         assert len(set(file_sizes)) == 1
#
#         return file_names, file_sizes[0]

