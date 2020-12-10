import numpy as np


def mask_index(warray):
    """Convert mask-array to a list of two-lists indicating the
    slice ends for all of the consecutively masked-out (i.e. ones marked by
    zero or False) indices.

    The output list may be empty if no such indices are found.

    >>> mask_index([True, True, False, False, False, True, False, True])
    [[2, 5], [6, 7]]
    >>> mask_index([False, False, False])
    [[0, 3]]
    >>> mask_index([True, True])
    []
    >>> mask_index([])
    []

    This is implemented using a simple state machine that scans over the input
    sequence.
    """
    # Overall container for all output chunks.
    res = []
    # Pad sentinel to the end of input.
    stream = enumerate(np.asarray(np.concatenate((warray, [1])), dtype=bool))
    # Container for the current chunk being found.
    thisres = []
    # A simple state machine: always seeking for a change ("edge") in the input
    # stream.
    state = "want_0"
    for i, w in stream:
        if w and (state == "want_0"):
            continue
        if w and (state == "want_1"):
            state = "want_0"
            thisres.append(i)
            res.append(thisres)
            thisres = []
            continue
        if (not w) and (state == "want_0"):
            state = "want_1"
            thisres.append(i)
            continue
        if (not w) and (state == "want_1"):
            continue
    return res


def dist(x, y) -> int:
    """Hamming distance between two equal-length sequences of bools."""
    return np.sum(np.asarray(x, dtype=bool) ^ np.asarray(y, dtype=bool))
