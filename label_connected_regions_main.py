import os
import cv2
import numpy as np

from skimage.morphology import label
from shutil import copyfile

# from PIL import Image


def main():
    origin_dir = '/data/public/rw/datasets/dsb2018/external_data/crowd_sourced/PSB_2015_ImageSize_400/Original_Images'
    label_dir = '/data/public/rw/datasets/dsb2018/external_data/crowd_sourced/PSB_2015_ImageSize_400/Nuclei_Segmentation/AutomatedMethodSegmentation'
    label_out_dir = '/data/public/rw/datasets/dsb2018/external_data/crowd_sourced/train'

    for root, dirs, filenames in os.walk(origin_dir):
        for fname in filenames:
            print(fname)
            file_id, ext = os.path.splitext(fname)
            images_dir = os.path.join(label_out_dir, file_id, 'images')
            os.makedirs(images_dir, exist_ok=True)

            try:
                img = cv2.imread(os.path.join(root, fname), cv2.IMREAD_COLOR)
                cv2.imwrite(os.path.join(label_out_dir, file_id, 'images', file_id + '.png'), img)
            except Exception as e:
                print(e)

    for root, dirs, filenames in os.walk(label_dir):
        for fname in filenames:
            print(fname)
            name, ext = os.path.splitext(fname)
            file_id = name.replace('_Binary', '')

            masks_dir = os.path.join(label_out_dir, file_id, 'masks')
            os.makedirs(masks_dir, exist_ok=True)

            img = cv2.imread(os.path.join(root, fname), cv2.IMREAD_COLOR)
            try:
                labels = label(img, connectivity=2)
                for i in range(1, labels.max() + 1):
                    label_img = (labels == i).astype(np.uint8) * 255

                    mask_fname = os.path.join(masks_dir, str(i)) + '.png'
                    cv2.imwrite(mask_fname, label_img)
            except Exception as e:
                print(e)



if __name__ == '__main__':
    main()
