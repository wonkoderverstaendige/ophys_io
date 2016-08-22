def check_data(data):
    """Sanity checks of records."""
    # Timestamps should increase monotonically in steps of 1024
    assert len(set(np.diff(data['timestamp']))) == 1 and np.diff(data['timestamp'][:2]) == 1024
    print('timestamps: ', data['timestamp'][0])

    # Number of samples in each record should be NUM_SAMPLES, or 1024
    assert len(set(data['n_samples'])) == 1 and data['n_samples'][0] == NUM_SAMPLES
    print('N samples: ', data['n_samples'][0])

    # should be byte pattern [0...8, 255]
    markers = set(map(str, data['rec_mark']))  # <- slow
    assert len(markers) == 1 and str(REC_MARKER) in markers
    print('record marker: ', data['rec_mark'][0])

    # should be zero, or there are multiple recordings in this file
    assert len(set(data['rec_num'])) == 1 and data['rec_num'][0] == 0
    print('Number recording: ', data['rec_num'][0])


def check_inputs(input_directories):
    """Check input directories for:
    - existence of all required files
    - matching files in all input directories
    - equal size of all files in a single directory
    - matching sampling rates
    - matching file size given record and header sizes
    - check for trailing zeros (check last for all 0, check at 2*distance each, then half distance to slow down)

    Args:
        List of input directories

    Returns:
        True if correct, False if errors occurred
    """
    # TODO: You know... do stuff.
    print(input_directories)