from os import path as op
from shutil import copyfile
import numpy as np
from .util import get_batch_size
import logging
from tqdm import trange

def batch_rereference_inplace(outfile, ref_file, precision='int16'):
    X = np.memmap(outfile, mode='r+', dtype=precision).reshape(-1, 64)
    batch_size, n_batches, batch_size_last = get_batch_size(X)
    mu = np.fromfile(ref_file, dtype='int16').reshape(-1, 1)
    print(mu.shape)
    try:
        for bn in trange(n_batches+1):
            bs = batch_size if bn < n_batches else batch_size_last
            X[bn*bs:(bn+1)*bs, :] -= mu[bn*bs:(bn+1)*bs]  # .astype(np.dtype(precision))
            # X[bn*bs:(bn+1)*bs, ch_idx_bad] = 0
    except BaseException as e:
        print(e)
    finally:
        del X

def make_ref_file(dat_file, out_file, n_channels=64, *args, **kwargs):
    # calculate mean over given array.
    print(dat_file, out_file)

    with open(dat_file, 'rb') as dat, open(out_file, 'wb+') as mu:
        dat_arr = np.memmap(dat, mode='r', dtype='int16').reshape(-1, n_channels)
        batch_reference(dat_arr, mu, *args, **kwargs)


def batch_reference(arr_like, file_out, good_chan_idx=None, bad_chan_idx=None, precision='int16'):
    # outfile = op.join(path_out, 'reference_{}.dat'.format(precision))
    batch_size, n_batches, batch_size_last = get_batch_size(arr_like)
    assert not (good_chan_idx is not None and bad_chan_idx is not None)
    if bad_chan_idx is not None:
        good_chan_idx = [c for c in range(arr_like.shape[1]) if c not in bad_chan_idx]

    print(good_chan_idx)
    # with open(outfile, 'wb+') as f:
    for bn in trange(n_batches+1):
        bs = batch_size if bn < n_batches else batch_size_last
        if good_chan_idx is None:
            batch = arr_like[bn * bs:(bn + 1) * bs, :]
        else:
            batch = arr_like[bn * bs:(bn + 1) * bs, :].take(good_chan_idx, axis=1)
        np.mean(batch, axis=1, dtype=precision).tofile(file_out)

def copy_as(src, dst):
    logging.info('Copying file {} to {}.'.format(src, dst))
    copyfile(src, dst)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', help='Dat file')
    parser.add_argument('-o', '--out', help='Directory to store reference file in')
    parser.add_argument('-b', '--bad_channels', type=int, nargs='+', help='Dead channel indices')
    parser.add_argument('-g', '--good_channels', type=int, nargs='+', help='Indices of channels to include')

    cli_args = parser.parse_args()
    make_ref_file(cli_args.input, cli_args.out, good_chan_idx=cli_args.good_channels, bad_chan_idx=cli_args.bad_channels)