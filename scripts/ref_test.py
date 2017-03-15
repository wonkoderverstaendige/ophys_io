from oio import referencing
import os.path as op

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-i')
    parser.add_argument('-r')
    parser.add_argument('-o')
    cli_args = parser.parse_args()
    dat_file = op.abspath(op.expanduser(cli_args.i))
    ref_file = op.abspath(op.expanduser(cli_args.r))
    # out_file = op.abspath(op.expanduser(cli_args.o))
    referencing.subtract_reference(dat_file, ref_file)