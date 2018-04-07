import os

import shutil

train_dir = '/data/public/rw/datasets/dsb2018/origin_ext_valid_kmeans'


def main():
    for dir in os.listdir(train_dir):
        dir_name = os.path.join(train_dir, dir)
        masks_dir = os.path.join(dir_name, 'masks')
        if os.path.exists(masks_dir):
            if len(os.listdir(masks_dir)) == 0:
                print(dir_name)
                shutil.rmtree(dir_name)


if __name__ == '__main__':
    main()
