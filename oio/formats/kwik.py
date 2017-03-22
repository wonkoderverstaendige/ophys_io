#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import xml.etree.ElementTree as ETree
from ..lib.tools import fext, path_content
import logging

FMT_NAME = 'Kwik'
FMT_FEXT = '.kwik'

logger = logging.getLogger(__name__)


def detect(base_path, pre_walk=None):
    """Checks for existence of a kwik formatted data set in the root directory.

    Args:
        base_path: Directory to search in.
        pre_walk: Tuple from previous path_content call (root, dirs, files)

    Returns:
        None if no data set found, else a dict of configuration data from settings.xml
    """
    root, dirs, files = path_content(base_path) if pre_walk is None else pre_walk

    for f in files:
        if fext(f) in ['.kwx', '.kwd', '.kwik']:
            return format_version(base_path, pre_walk)


def format_version(base_path, pre_walk=None):
    root, dirs, files = path_content(base_path) if pre_walk is None else pre_walk

    if "settings.xml" in files:
        xml_root = ETree.parse(os.path.join(base_path, 'settings.xml'))
        version = xml_root.findall("INFO/VERSION")[0].text
        if len(version):
            return "{}_v{}".format(FMT_NAME, version if version else '???')


def config(base_path, *args, **kwargs):
    logger.debug('Returning "empty" config dict')
    return {'HEADER': {'sampling_rate': None},
            'INFO': None,
            'SIGNALCHAIN': None,
            'FPGA_NODE': None,
            'AUDIO': None}


def fill_buffer(target, buffer, offset, count, *args, **kwargs):
    raise NotImplementedError("Can't Kwik yet.")
