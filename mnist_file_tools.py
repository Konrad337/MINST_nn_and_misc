import numpy as np
import itertools as it
import struct


def get_bytes(file, bytes=4):
    if bytes is 4:
        return int(struct.unpack('>i', file.read(4))[0])
    elif bytes is 1:
        return ord(file.read(1))
# Get integer value of given bytes


def get_input_layer(file, rows, columns):
    layer = np.zeros((rows, columns), dtype=int)
    for i, j in it.product(range(rows), range(columns)):
        layer[i, j] = get_bytes(file, 1)
    return layer
# 2-d array of one digit


def get_label(file):
    return get_bytes(file, 1)
# next label
