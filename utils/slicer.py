import numpy as np
from numpy.random import default_rng

from utils.preprocess import recs2dataset

rng = default_rng()


def verbose_split(train_idx, test_idx):
    i = 0
    for tr, ts in zip(train_idx, test_idx):
        i += 1

        partial_load = sum(len(t) for t in tr)
        rec_len = partial_load + sum(len(t) for t in ts)
        print(f"Recording {i}:")
        print(f"\tPartial load:\t{partial_load / rec_len:02},\t{partial_load}/{rec_len} frames")

    total_load = sum(sum(len(t) for t in tr) for tr in train_idx)
    total_length = total_load + sum(sum(len(t) for t in ts) for ts in test_idx)
    print(f"Obtained load:\t{total_load / total_length:02},\t{total_load}/{total_length} frames")


def train_split_no_cut(recording_lengths, train_load, labels, verbose=True):
    """
    Perform test-train split of a list of recordings without cutting recordings, only assigning recordings
    to different sets in a label-balanced way
    Args:
        recording_lengths: List of int, each one of them represents the length of a recording file
        train_load: float, Desired train load in [0.5, 1)
        labels: List of int representing the label assigned to each recording to take into account for splitting
        verbose: Verbose output to verify the split

    Returns:
        train_idx, test_idx
        The indices for the train and test indices as two nested lists of range objects

    """

    if not (0.5 <= train_load < 1):
        raise ValueError(f"Invalid value for train_load={train_load}.\n"
                         f"It must lie within [0.5 , 1).")

    label_lengths = {}
    for rec_len, lbl in zip(recording_lengths, labels):
        try:
            label_lengths[lbl].append(rec_len)
        except KeyError:
            label_lengths[lbl] = [rec_len]
    total_loads = {k: round(sum(v) * train_load) for k, v in label_lengths.items()}

    train_idx = []
    test_idx = []
    acc_lengths = {k: 0 for k in total_loads}
    for r_len, lbl in zip(recording_lengths, labels):
        if acc_lengths[lbl] < total_loads[lbl]:
            train_idx.append([range(0, r_len)])
            test_idx.append([range(0, 0)])
        else:
            train_idx.append([range(0, 0)])
            test_idx.append([range(0, r_len)])
        acc_lengths[lbl] += r_len

    if verbose:
        print(f"Required load:\t{train_load}")
        verbose_split(train_idx, test_idx)

    return train_idx, test_idx


def train_split_singlecut(recording_lengths, train_load, beta=10.0, verbose=True):
    """
    Perform test-train split of a list of recordings through a single random cut on every recording
    Args:
        recording_lengths: List of int, each one of them represents the length of a recording file
        train_load: float, Desired train load in [0.5, 1)
        beta: Parameter of the beta distribution. The bigger, the less standard deviation over the random cuts.
            If None, it divides every measurement deterministically according to train_load
        verbose: Verbose output to verify the split

    Returns:
        train_idx, test_idx
        The indices for the train and test indices as two nested lists of range objects

    """

    if not (0.5 <= train_load < 1):
        raise ValueError(f"Invalid value for train_load={train_load}.\n"
                         f"It must lie within [0.5 , 1).")

    if beta is None:
        deterministic = True
        pass
    elif beta < 1:
        raise ValueError(f"Invalid value for beta={beta}.\n"
                         f"It must be equal or greater than one to enforce unimodality.")
    else:
        deterministic = False

    rec_lengths = np.array(recording_lengths)
    n_recs = rec_lengths.size
    total_length = rec_lengths.sum()

    # Use the longest recording to even out the obtained load
    i_max = rec_lengths.argmax()
    max_length = rec_lengths[i_max]
    max_out_idx = np.arange(n_recs) != i_max

    total_load = np.round(total_length * train_load)

    max_out_loads = None
    max_load = -1
    if deterministic:
        max_out_loads = np.round(train_load * rec_lengths[max_out_idx])
        max_load = total_load - max_out_loads.sum()
    else:   # Use a beta distribution to generate random loads with rejection
        while not 0 < max_load < max_length:
            alpha = beta * train_load / (1 - train_load)
            beta_loads = rng.beta(alpha, beta, size=n_recs - 1)
            max_out_loads = np.round(beta_loads * rec_lengths[max_out_idx])
            max_load = total_load - max_out_loads.sum()

    partial_loads = np.zeros(n_recs, dtype=np.int32)
    partial_loads[max_out_idx] = max_out_loads
    partial_loads[i_max] = max_load

    head_tail = rng.binomial(1, 0.5, size=n_recs)   # Choose randomly the head or the tail of the chunk

    train_idx = []
    test_idx = []

    for head_train, partial_load, rec_len in zip(head_tail, partial_loads, rec_lengths):
        if head_train:
            train_idx.append([range(0, partial_load)])
            test_idx.append([range(partial_load, rec_len)])
        else:
            train_idx.append([range(rec_len - partial_load, rec_len)])
            test_idx.append([range(0, rec_len - partial_load)])

    if verbose:
        print(f"Longest recording is {i_max + 1}, {rec_lengths[i_max]} frames")
        print(f"Required load:\t{train_load}")
        verbose_split(train_idx, test_idx)

    return train_idx, test_idx


def train_split_doublecut(recording_lengths, train_load, min_len=1, stride=1, verbose=True):
    """
    Perform test-train split of a list of recordings through a double cut of fixed length
    at a random position of every recording
    Args:
        recording_lengths: List of int, each one of them represents the length of a recording file
        train_load: float, Desired train load in [0.5, 1)
        min_len: int, Minimum length of a chunk, important if the data will be further sliced
        stride: int, Stride to consider when choosing the random position of the cuts
        verbose: Verbose output to verify the split

    Returns:
        train_idx, test_idx
        The indices for the train and test indices as two nested lists of range objects

    """

    if not (0.5 <= train_load < 1):
        raise ValueError(f"Invalid value for train_load={train_load}.\n"
                         f"It must lie within [0.5 , 1).")

    rec_lengths = np.array(recording_lengths)
    n_recs = rec_lengths.size
    total_length = rec_lengths.sum()

    # Use the longest recording to even out the obtained load
    i_max = rec_lengths.argmax()
    max_out_idx = np.arange(n_recs) != i_max

    total_load = np.round(total_length * train_load)

    max_out_loads = np.round(train_load * rec_lengths[max_out_idx])
    max_load = total_load - max_out_loads.sum()

    partial_loads = np.zeros(n_recs, dtype=np.int32)
    partial_loads[max_out_idx] = max_out_loads
    partial_loads[i_max] = max_load

    train_idx = []
    test_idx = []

    for partial_load, rec_len in zip(partial_loads, rec_lengths):

        trimmed_length = rec_len - partial_load

        offset_choices = [0, trimmed_length]                                                # head and tail offsets
        offset_choices.extend(range(min_len, trimmed_length - min_len, stride))             # centered train chunk
        offset_choices.extend(range(trimmed_length + min_len, rec_len - min_len, stride))   # centered test chunk

        offset = rng.choice(offset_choices)
        second_cut = (offset + partial_load) % rec_len

        if offset == 0:                                                 # head train chunk
            train_idx.append([range(offset, partial_load)])
            test_idx.append([range(partial_load, rec_len)])
        elif offset == trimmed_length:                                  # tail train chunk
            train_idx.append([range(offset, rec_len)])
            test_idx.append([range(0, offset)])
        elif offset < second_cut:                                       # centered train chunk
            train_idx.append([range(offset, second_cut)])
            test_idx.append([range(0, offset),
                             range(second_cut, rec_len)])
        else:                                                           # centered test chunk
            train_idx.append([range(0, second_cut),
                              range(offset, rec_len)])
            test_idx.append([range(second_cut, offset)])

    if verbose:
        print(f"Longest recording is {i_max + 1}, {rec_lengths[i_max]} frames")
        print(f"Required load:\t{train_load}")
        verbose_split(train_idx, test_idx)

    return train_idx, test_idx


def serialize_ranges(ranges):
    if isinstance(ranges, range):
        return {"start": ranges.start, "stop": ranges.stop}
    elif isinstance(ranges, (tuple, list)):
        return [serialize_ranges(r) for r in ranges]
    else:
        raise TypeError(f"Unexpected class {type(ranges)}")


def deserialize_ranges(ranges):
    if isinstance(ranges, dict):
        return range(ranges["start"], ranges["stop"])
    elif isinstance(ranges, (tuple, list)):
        return [deserialize_ranges(r) for r in ranges]
    else:
        raise TypeError(f"Unexpected class {type(ranges)}")


def index_ranges(ranges, recordings):
    lut_date = {r.date: i for i, r in enumerate(recordings)}
    new_ranges = []
    for date, r in ranges.items():
        new_ranges.append((lut_date[date], deserialize_ranges(r)))
    return new_ranges


def train_test_slice(recordings, time_length, time_hop, train_load, split="single", beta=20.0,
                     verbose=True, merge=False, return_segments=False):
    """Slice recordings into train and test datasets

    Args:
        recordings: Sequence of Dataset objects holding the preprocessed recordings
        time_length: Length of the data examples along the time axis
        time_hop: Hop size between the slices across the time axis
        train_load: float, Desired train load in [0.5, 1)
        return_segments: Return serialized segments
        verbose: Verbose output to verify the split
        merge: If True, merge slices into a new lazy dataset, otherwise return a flat indexed slice list
        split: Split variant, either "deterministic", single" or "double" or None for no train-test split
        beta: float, beta value for the random single split

    Returns: train_idx, test_idx
        Train and test Dataset objects or slice lists, depending on 'merge'

    """

    rec_lens = [r.sizes["time"] for r in recordings]
    rec_labels = [r.label.item() for r in recordings]

    def slice_indices(indices):
        sliced_indices = []
        for i, rec_segments in enumerate(indices):
            for range_seg in rec_segments:
                for offset in range(0, len(range_seg) - time_length + 1, time_hop):
                    sliced_indices.append((i, range_seg[offset:offset+time_length]))
        return sliced_indices

    train_splits = {"deterministic": lambda rcl, tld: train_split_singlecut(rcl, tld, beta=None, verbose=verbose),
                    "no-cut": lambda rcl, tld: train_split_no_cut(rcl, tld, labels=rec_labels, verbose=verbose),
                    "single": lambda rcl, tld: train_split_singlecut(rcl, tld, beta=beta, verbose=verbose),
                    "double": lambda rcl, tld: train_split_doublecut(rcl, tld, min_len=time_length,
                                                                     stride=time_hop, verbose=verbose)}

    if split is None:
        full_idx = [[range(rl)] for rl in rec_lens]
        full_slices = slice_indices(full_idx)
        if merge:
            return recs2dataset(recordings, full_slices)
        else:
            return full_slices
    else:
        if isinstance(split, dict):
            return_segments = False
            rec_dates = [r.date for r in recordings]
            train_split = split['train']
            test_split = split['test']

            train_idx = [deserialize_ranges(train_split[d]) for d in rec_dates]
            test_idx = [deserialize_ranges(test_split[d]) for d in rec_dates]
        else:
            try:
                train_idx, test_idx = train_splits[split](rec_lens, train_load)
            except KeyError as ke:
                ts_keys = tuple(train_splits.keys())
                raise ValueError(f"Unrecognized split argument '{split}'."
                                 f" Argument must belong to {ts_keys}") from ke

        train_sliced = slice_indices(train_idx)
        test_sliced = slice_indices(test_idx)

        if merge:
            train_ds = recs2dataset(recordings, train_sliced)
            test_ds = recs2dataset(recordings, test_sliced)
            ret = [train_ds, test_ds]
        else:
            ret = [train_sliced, test_sliced]

        if return_segments:
            serial_segments = {"train": {r.date: s for r, s in zip(recordings, serialize_ranges(train_idx))},
                               "test":  {r.date: s for r, s in zip(recordings, serialize_ranges(test_idx))}}
            ret.append(serial_segments)

        return ret
