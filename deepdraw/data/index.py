from os.path import join

from .unpack import unpack_all


def build(path):
    """Builds index file from data directory.

    Index file is saved to path/index.csv and has the following format:
    {key_id},{word},{offset}

    Args:
        path (str): path to data directory
    """
    index = join(path, 'index.csv')
    with open(index, 'w') as f:
        f.write("key_id,word,offset\n")
        for drawing in unpack_all(path):
            id = drawing['key_id']
            word = drawing['word']
            offset = drawing['offset']
            line = "{id},{word},{offset}\n".format(id=id,
                                                   word=word,
                                                   offset=offset)
            f.write(line)


if __name__ == '__main__':
    import sys
    if len(sys.argv) != 2:
        print("Usage: build-index DATA_DIRECTORY")
        sys.exit(1)
    else:
        path = sys.argv[1]
        build(path)
