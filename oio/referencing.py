from os import path as op
from shutil import copyfile
import numpy as np
from .util import get_batch_size
import logging
from tqdm import trange

def ref(dat_file, ref_file=None, n_channels=64, *args, **kwargs):
    if ref_file==None:
        ref_file = make_ref_file(dat_file, n_channels=n_channels, *args, **kwargs)

    subtract_reference(dat_file, ref_file, n_channels=n_channels, *args, **kwargs)

def subtract_reference(dat_file, ref_file, precision='single', inplace=False, n_channels=64):
    # if inplace, just overwrite, in_file, else, make copy of in_file
    # FIXME: Also memmap the reference file? Should be small even for long recordings...
    # FIXME: If not inplace, the file should be opened read only!

    with open(dat_file, 'r+b') as dat, open(ref_file, 'rb') as mu:
        print(dat)
        dat_arr = np.memmap(dat, dtype='int16').reshape(-1, n_channels)


        mu_arr = np.fromfile(mu, dtype=precision).reshape(-1, 1)
        assert (dat_arr.shape[0] == mu_arr.shape[0])

    batch_size, n_batches, batch_size_last = get_batch_size(dat_arr)
    fname, ext = op.splitext(dat_file)

    try:
        if inplace:
            out_arr = dat_arr
        else:
            out_arr = np.memmap(fname + '_mean_referenced' + ext, mode='w+', dtype=dat_arr.dtype, shape=dat_arr.shape)

        for bn in trange(n_batches+1):
            bs = batch_size if bn < n_batches else batch_size_last
            if inplace:
                out_arr[bn*bs:(bn+1)*bs, :] -= mu_arr[bn*bs:(bn+1)*bs]  # .astype(np.dtype(precision))
            else:
                out_arr[bn*bs:(bn+1)*bs, :] = dat_arr[bn*bs:(bn+1)*bs, :] - mu_arr[bn*bs:(bn+1)*bs]

            # Zero dead channels?
            # out_arr[bn*bs:(bn+1)*bs, ch_idx_bad] = 0
    except BaseException as e:
        print(e)
    finally:
        del dat_arr

def make_ref_file(dat_file, n_channels, out_file=None, *args, **kwargs):
    # calculate mean over given array.
    if out_file is None:
        fname, ext = op.splitext(dat_file)
        out_file = fname + '_reference' + ext
    print(dat_file, out_file)

    with open(dat_file, 'rb') as dat, open(out_file, 'wb+') as mu:
        dat_arr = np.memmap(dat, mode='r', dtype='int16').reshape(-1, n_channels)
        _batch_reference(dat_arr, mu, *args, **kwargs)
    return out_file

def _batch_reference(arr_like, out_file, good_chan_idx=None, bad_chan_idx=None, precision='float32'):
    # outfile = op.join(path_out, 'reference_{}.dat'.format(precision))
    batch_size, n_batches, batch_size_last = get_batch_size(arr_like)
    assert not (good_chan_idx is not None and bad_chan_idx is not None)
    if bad_chan_idx is not None:
        good_chan_idx = [c for c in range(arr_like.shape[1]) if c not in bad_chan_idx]

    for bn in trange(n_batches+1):
        bs = batch_size if bn < n_batches else batch_size_last
        if good_chan_idx is None:
            batch = arr_like[bn * bs:(bn + 1) * bs, :]
        else:
            batch = arr_like[bn * bs:(bn + 1) * bs, :].take(good_chan_idx, axis=1)
        np.mean(batch, axis=1, dtype=precision).tofile(out_file)

def copy_as(src, dst):
    logging.info('Copying file {} to {}.'.format(src, dst))
    copyfile(src, dst)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', help='Dat file')
    parser.add_argument('-o', '--out', help='Directory to store reference file in', default='.')
    parser.add_argument('-r', '--reference', help='Path to reference file, if already at hand.')
    parser.add_argument('-b', '--bad_channels', type=int, nargs='+', help='Dead channel indices')
    parser.add_argument('-g', '--good_channels', type=int, nargs='+', help='Indices of channels to include')
    parser.add_argument('-C', '--channel_count', type=int, help='Number of channels in input file.', default=64)

    cli_args = parser.parse_args()
    make_ref_file(cli_args.input,
                  cli_args.channel_count,
                  good_chan_idx=cli_args.good_channels,
                  bad_chan_idx=cli_args.bad_channels)