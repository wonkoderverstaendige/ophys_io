#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import re
from termcolor import colored
import logging

logger = logging.getLogger(__name__)

def fmt_size(num, unit='B', si=True, sep=' ', col=False, pad=0):
    colors = {"k": "blue", "M": "green", "G": "red", "T": "cyan",
              "Ki": "blue", "Mi": "green", "Gi": "red", "Ti": "cyan"}
    if si:
        prefixes = ['', 'k', 'M', 'G', 'T', 'P', 'E']
    else:
        prefixes = ['', 'Ki', 'Mi', 'Gi', 'Ti', 'Pi', 'Ei']
    
    divisor = 1000 if si else 1024
    for prefix in prefixes:
        if abs(num) < divisor:
            if prefix:
                prefix = colored(prefix, colors[prefix]) if col else prefix
                return "{:5.1f}{}{}{}".format(num, sep, prefix, unit, pad=pad-6)
            else:
                return "{:5.0f}{}{}{} ".format(num, sep, prefix, unit, pad=pad-6)
        num /= divisor


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


def fext(fname):
    """Grabs the file extension of a file.

    Args:
        fname: File name.

    Returns:
        String with file extension. Empty string, if file has no extensions.

    Raises:
        IOError if file does not exist or can not be accessed.
    """
    return os.path.splitext(fname)[1]


def full_path(path):
    """Return full path of a potentially relative path, including ~ expansion.

    Args:
        Path

    Returns:
        Absolute(Expanduser(Path))
    """
    return os.path.abspath(os.path.expanduser(path))

def path_content(path):
    """Gathers root and first level content of a directory.

    Args:
        path: Relative or absolute path to a directory.

    Returns:
        A tuple containing the root path, the directories and the files
        contained in the root directory.

        (path, dir_names, file_names)
    """
    path = full_path(path)
    assert(os.path.exists(path))
    if os.path.isdir((path)):
        return next(os.walk(path))
    else:
        return os.path.basename(path), [], [path]

def dir_size(path):
    """Calculate size of directory including all subdirectories and files

    Args:
        path: Relative or absolute path.

    Returns:
        Integer value of size in Bytes.
    """
    logger.debug('dir_size path: {}'.format(path))
    assert os.path.exists(path)
    if not os.path.isdir(path):
        return os.path.getsize(path)

    total_size = 0
    for root, dirs, files in os.walk(path):
        for f in files:
            fp = os.path.join(root, f)
            try:
                total_size += os.path.getsize(fp)
            except OSError:
                # symbolic links cause issues
                pass
    return total_size

def terminal_size():
    """Get size of currently used terminal. In many cases this is inaccurate.

    Returns:
        Tuple of width, height.

    Raises:
        Unknown error when not run from a terminal.
    """
    # return map(int, os.popen('stty size', 'r').read().split())
    # Python 3.3+
    ts = os.get_terminal_size()
    return ts.lines, ts.columns

def find_getch():
    """Helper to wait for a single character press, instead of having to use raw_input() requiring Enter
    to be pressed. Should work on all OS.

    Returns:
        Function that works as blocking single character input without prompt.
    """
    # FIXME: Find where I took this piece of code from... and attribute. SO perhaps?
    try:
        import termios
    except ImportError:
        # Non-POSIX. Return msvcrt's (Windows') getch.
        import msvcrt
        return msvcrt.getch

    # POSIX system. Create and return a getch that manipulates the tty.
    import sys, tty
    def _getch():
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            ch = sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        return ch

    return _getch


ansi_escape = re.compile(r'\x1b[^m]*m')
def strip_ansi(string):
    """Remove the ANSI codes (e.g. color and additional formatting) from a string.

    Args:
        string: A string potentially containing ANSI escape codes.

    Returns:
        String with ANSI escape codes removed.
    """
    return ansi_escape.sub('', string)

if __name__ == "__main__":
    pass
