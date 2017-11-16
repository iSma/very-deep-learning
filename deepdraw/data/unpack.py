import glob
import struct
from struct import unpack
from os.path import basename, join, splitext

"""
Binary format unpacking adapted from:
https://github.com/googlecreativelab/quickdraw-dataset/blob/master/examples/binary_file_parser.py
"""


def unpack_drawing(f):
    """Unpack next drawing in file"""
    offset = f.tell()
    key_id, = unpack('Q', f.read(8))
    countrycode, = unpack('2s', f.read(2))
    recognized, = unpack('b', f.read(1))
    timestamp, = unpack('I', f.read(4))
    n_strokes, = unpack('H', f.read(2))
    drawing = []
    for i in range(n_strokes):
        n_points, = unpack('H', f.read(2))
        fmt = str(n_points) + 'B'
        x = unpack(fmt, f.read(n_points))
        y = unpack(fmt, f.read(n_points))
        drawing.append((x, y))

    return {
        'key_id': key_id,
        'offset': offset,
        'countrycode': countrycode.decode(),
        'recognized': recognized,
        'timestamp': timestamp,
        'drawing': drawing
    }


def unpack_file(path):
    """Unpacks all drawings in file"""
    word = basename(path)
    word = splitext(word)[0]

    with open(path, 'rb') as f:
        while True:
            try:
                drawing = unpack_drawing(f)
                drawing['word'] = word
                yield drawing
            except struct.error:
                break


def unpack_all(path):
    """Unpacks all drawings from the given directory"""
    path = join(path, '*.bin')
    for path in glob.glob(path):
        yield from unpack_file(path)


def load(path, word, offset):
    """Loads a specific drawing

    Args:
        path (str): path to data directory
        word (str): category of drawing (airplane, banana, ...)
        offset (int): byte offset inside file
    """
    path = join(path, word + '.bin')
    with open(path, 'rb') as f:
        f.seek(offset)
        drawing = unpack_drawing(f)
        drawing['word'] = word

        return drawing
