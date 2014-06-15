#!/usr/bin/env python

import argparse
import os
import sys
from glob import glob
import logging
import re

def last_increment(target_dir, basename=None):
    """Find the highest increment using the filenaming scheme [xxx_subject]
    """
    try:
        basename = basename if basename is not None else os.path.split(os.path.abspath(target_dir))[-1]
        log.debug('Basename parent directory: %s' % basename)
         
        dircontent = glob(os.path.join(target_dir, '[0-9][0-9][0-9]_m'+basename))
        print dircontent
        filenames = [os.path.basename(fname) for fname in dircontent] 
        return int(sorted(filenames)[-1][0:3])
    except (ValueError, IndexError):
        return 0
            
def dir_list(target_dir, basename=None):
    basename = basename if basename is not None else os.path.split(os.path.abspath(target_dir))[-1]
    subdirs = sorted([os.path.basename(dir_name) for dir_name in glob(os.path.join(target_dir, basename+'*')) if os.path.isdir(dir_name)])
    print subdirs
    return subdirs

def rename(target_dir, old_games, new_names):
    assert len(old_names) == len(new_names)
    for name_pair in zip(old_names, new_names):
        old_name = os.path.join(target_dir, name_pair[0])
        new_name = os.path.join(target_dir, name_pair[1])
        try:
            assert not os.path.exists(new_name)
            log.debug("Renaming {0} to {1}".format(old_name, new_name))
            os.rename(old_name, new_name)
        except AssertionError:
            log.error("Name {0} already exists! Couldn't rename {1}".format(new_name, old_name))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Rename folders created by open ephys to the names using incrementing IDs', \
            prog='rename.py')
    parser.add_argument('-t', '--target', help='Target directory containing files to rename', required=True) 
    parser.add_argument('-i', '--input', help='File containing rename list')
    parser.add_argument('-s', '--start', help='Starting index number', type=int, default=1)
    parser.add_argument('-l', '--leading', help='Leading zeros, larger than zero', default=3, type=int)
    parser.add_argument('-p', '--post', help='Postfix string', default='_m2541')
    parser.add_argument('-v', '--verbose', help='Show debugging messages', action='store_const', const=logging.DEBUG, \
            default=logging.INFO)

    args = parser.parse_args()
#    verbosity = logging.DEBUG if args.verbose else logging.INFO 
#   logging.basicConfig(level=args.verbose, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logging.basicConfig(level=args.verbose, format='%(levelname)s - %(message)s')

    log = logging.getLogger(__name__)

    # find name of animal from parent directory name
    parent_dir = os.path.split(args.target)[-1]
    log.debug('Parent directory: {0}'.format(parent_dir))
    basename = re.search('\d+', parent_dir).group(0) if re.search('\d+', parent_dir) else None
    log.debug(basename)

    #  find last increment
    last_inc = last_increment(args.target, basename=basename)
    log.debug('Last increment: %d' % last_inc)

    # find list of yet-to-be-renamed folders
    old_names = dir_list(args.target, basename=basename)
 
    # rename those folders
    start = 0 if last_inc == 0 else last_inc+1
    end = start + len(old_names)
    log.debug("Range: {0}, {1}".format(start, end))
    new_names = ['{0:03d}_{1}'.format(inc, parent_dir) for inc in range(start, end)]
    rename(args.target, old_names, new_names)

    # DONE
 
