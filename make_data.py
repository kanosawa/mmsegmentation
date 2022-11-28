import os
from glob import glob


def main():
    paths = glob('data/live2d/train/*.png')
    with open('data/live2d/train.txt', 'w') as f:
        for path in paths:
            basename, _ = os.path.splitext(os.path.basename(path))
            f.write(basename + '\n')


if __name__ == '__main__':
    main()
