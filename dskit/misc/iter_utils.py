from typing import Sequence


def circular_pairs(seq: Sequence):
    n = len(seq)
    for i in range(n):
        yield seq[i], seq[(i + 1) % n]
