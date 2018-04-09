import os

import shutil

src_dir = '/data/public/rw/datasets/dsb2018/external_data/crowd_sourced/train_gray_full'
dest_dir = '/data/public/rw/datasets/dsb2018/external_data/crowd_sourced/train_gray'


def main():
    for idx, dir in enumerate(os.listdir(src_dir)[100:200]):
        src_dir_name = os.path.join(src_dir, dir)
        dest_dir_name = os.path.join(dest_dir, dir)
        print(idx, 'copy', src_dir_name, 'to', dest_dir_name)
        shutil.copytree(src_dir_name, dest_dir_name)


if __name__ == '__main__':
    main()
