#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Mutable buffer/array that can be shared between multiple processes.
Inspired by https://github.com/belevtsoff/rdaclient.py

* `Buffer`: The buffer
* `datatypes`: supported datatypes
* `BufferHeader`: header structure containing metadata
* `BufferError`: error definitions

"""

from multiprocessing import Array
import ctypes as ct
import logging

import numpy as np
logger = logging.getLogger("SharedBuffer")


class SharedBuffer:
    """One-dimensional buffer with homogeneous elements.

    The buffer can be used simultaneously by multiple processes, because
    both data and metadata are stored in a single sharedctypes byte array.
    First, the buffer object is created and initialized in one of the
    processes. Second, its raw array is shared with others. Third, those
    processes create their own Buffer objects and initialize them so that
    all point to the same shared raw array.
    """
    def __init__(self):
        self.initialized = False
        self.n_channels = None
        self.n_samples = None
        self.raw = None
        self.buffer = None
        self.buffer_hdr = None
        self.buffer_size = None
        self.np_type = None

    def __str__(self):
        return self.buffer[:self.buffer_size].__str__() + '\n'

    # def __getattr__(self, item):
    #     """Overload to prevent access to the buffer attributes before
    #     initialization is complete.
    #     """
    #     if self.initialized:
    #         return object.__getattribute__(self, item)
    #     else:
    #         raise BufferError(1)

    # -------------------------------------------------------------------------
    # PROPERTIES

    # read only attributes
    # is_initialized = property(lambda self: self.__initialized, None, None,
    #                           'Indicates whether the buffer is initialized, read only (bool)')
    # raw = property(lambda self: self.__raw, None, None,
    #                'Raw buffer array, read only (sharedctypes, char)')
    # nChannels = property(lambda self: self.__nChannels, None, None,
    #                      'Dimensionality of array in channels, read only (int)')
    # nSamples = property(lambda self: self.__nSamples, None, None,
    #                     'Dimensionality of array in samples, read only (int)')
    # bufSize = property(lambda self: self.__bufSize, None, None,
    #                    'Buffer size, read only (int)')
    # nptype = property(lambda self: self.__nptype, None, None,
    #                   'The type of the data in the buffer, read only (string)')

    # -------------------------------------------------------------------------

    def initialize(self, n_channels, n_samples, np_dtype='float32'):
        """Initializes the buffer with a new array."""
        logger.debug('Initializing {}x{} {} buffer.'.format(n_channels, n_samples, np_dtype))

        # check parameters
        if n_channels < 1 or n_samples < 1:
            logger.error('n_channels and n_samples must be a positive integer')
            raise SharedBufferError(1)

        size_bytes = ct.sizeof(SharedBufferHeader) + n_samples * n_channels * np.dtype(np_dtype).itemsize
        raw = Array('c', size_bytes)
        hdr = SharedBufferHeader.from_buffer(raw.get_obj())

        hdr.bufSizeBytes = size_bytes - ct.sizeof(SharedBufferHeader)
        hdr.dataType = DataTypes.get_code(np_dtype)
        hdr.nChannels = n_channels
        hdr.nSamples = n_samples
        hdr.position = 0

        self.initialize_from_raw(raw.get_obj())

    def initialize_from_raw(self, raw):
        """Initializes the buffer with the compatible external raw array.
        All the metadata will be read from the header region of the array.
        """
        logger.debug('Initializing from raw buffer {}'.format(raw))
        self.raw = raw
        hdr = SharedBufferHeader.from_buffer(self.raw)

        # data type
        self.np_type = DataTypes.get_type(hdr.dataType)

        buf_offset = ct.sizeof(hdr)
        buf_flat_size = hdr.bufSizeBytes // np.dtype(self.np_type).itemsize

        # create numpy view object pointing to the raw array
        self.buffer_hdr = hdr
        self.buffer = np.frombuffer(self.raw, self.np_type, buf_flat_size, buf_offset) \
            .reshape((hdr.nChannels, -1))

        # helper variables
        self.n_channels = hdr.nChannels
        self.n_samples = hdr.nSamples
        self.buffer_size = len(self.buffer)

        # Ready?
        logger.debug('Buffer with shape {}, dtype: {}'.format(self.buffer.shape, self.buffer.dtype))
        self.initialized = True

    def __write_buffer(self, data, start, end=None, channel=None):
        """Writes data to buffer."""
        # roll array
        # overwrite old section
        if end is None:
            end = start+data.shape[1]
        if channel is None:
            self.buffer[:, start:end] = data
        else:
            self.buffer[channel, start:end] = data

    def __read_buffer(self, start, end):
        """Reads data from buffer, returning view into numpy array"""
        av_error = self.check_availability(start, end)
        if not av_error:
            return self.buffer[:, start:end]
        else:
            raise SharedBufferError(av_error)

    def get_data(self, start=0, end=None, wprotect=False):
        end = end if end is not None else self.buffer.shape[1]
        data = self.__read_buffer(start, end)
        data.setflags(write=not wprotect)
        return data

    def put_data(self, data, start=0, channel=None):
        """Put data into the buffer. Either a single channel, or
        overwrite the full array.
        """
        # FIXME: Detect which case to use based on the shape of the data
        # should update the whole array at once
        logger.debug('Putting data of shape {} at start {}'.format(data.shape, start))
        if len(data.shape) != 1:
            if data.shape != self.buffer.shape:
                raise SharedBufferError(4)
        else:
            data = data.reshape(1, -1)
            # data.shape = (1, len(data))
            # if channel >= self.buffer or channel < 0:
            #     raise SharedBufferError(4)

        end = start + data.shape[1]
        self.__write_buffer(data, start, end, channel)

    def check_availability(self, start, end):
        """Checks whether the requested data samples are available.

        Parameters
        ----------
        start : int
            first sample index (included)
        end : int
            last samples index (excluded)

        Returns
        -------
        0
            if the data is available and already in the buffer
        1
            if the data is available but needs to be read in
        2
            if data is partially unavailable
        3
            if data is completely unavailable


        """
        if start < end:
            return 0
        else:
            print(self.buffer.shape)
            return
        # if sampleStart < 0 or sampleEnd <= 0:
        #     return 5
        # if sampleEnd > self.nSamplesWritten:
        #     return 3  # data is not ready
        # if (self.nSamplesWritten - sampleStart) > self.bufSize:
        #     return 2  # data is already erased
        #
        # return 0


class DataTypes:
    """A helper class to interpret the type code read from buffer header.
    To add new supported data types, add them to the 'type' dictionary
    """
    types = {0: 'float32',
             1: 'int16'}
    types_rev = {v: k for k, v in types.items()}

    @classmethod
    def get_code(cls, np_dtype):
        """Gets buffer type code given numpy datatype

        Parameters
        ----------
        np_dtype : string
            numpy datatype (e.g. 'float32')
        """
        return cls.types_rev[np_dtype]

    @classmethod
    def get_type(cls, code):
        """Gets numpy data type given a buffer type code

        Parameters
        ----------
        code : int
            type code (e.g. 0)
        """
        return cls.types[code]


class SharedBufferHeader(ct.Structure):
    """A ctypes structure describing the buffer header

    Attributes
    ----------
    bufSizeBytes : c_ulong
        size of the buffer in bytes, excluding header and pocket
    dataType : c_uint
        typecode of the data stored in the buffer
    nChannels : c_ulong
        sample dimensionality
    nSamples : c_ulong
        size of the buffer in samples
    position : c_ulong
        position in the data in samples
    """
    _pack_ = 1
    _fields_ = [('bufSizeBytes', ct.c_ulong),
                ('dataType', ct.c_uint),
                ('nChannels', ct.c_ulong),
                ('nSamples', ct.c_ulong),
                ('position', ct.c_ulong)]


class SharedBufferError(Exception):
    """Represents different types of buffer errors"""

    def __init__(self, code):
        """Initializes a BufferError with given error code

        Parameters
        ----------
        code : int
            error code
        """
        self.code = code

    def __str__(self):
        """Prints the error"""
        if self.code == 1:
            return 'buffer is not initialized (error {})'.format(self.code)
        elif self.code in [2, 3]:
            return 'unable to get indices (error {})'.format(self.code)
        elif self.code == 4:
            return 'writing incompatible data (error {})'.format(self.code)
        elif self.code == 5:
            return 'negative index (error {})'.format(self.code)
        else:
            return '(error {})'.format(self.code)


if __name__ == '__main__':
    pass
