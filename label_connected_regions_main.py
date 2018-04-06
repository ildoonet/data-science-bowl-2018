import os
import cv2
import numpy as np

from skimage.morphology import label
from shutil import copyfile





def main():
    origin_dir = '/data/public/rw/datasets/dsb2018/external_data/crowd_sourced/PSB_2015_ImageSize_400/Original_Images'
    label_dir = '/data/public/rw/datasets/dsb2018/external_data/crowd_sourced/PSB_2015_ImageSize_400/Nuclei_Segmentation/AutomatedMethodSegmentation'
    label_out_dir = '/data/public/rw/datasets/dsb2018/external_data/crowd_sourced/train'
    for root, dirs, filenames in os.walk(label_dir):
        for fname in filenames:
            print(fname)
            name, ext = os.path.splitext(fname)
            file_id = name.replace('_Binary', '')

            images_dir = os.path.join(label_out_dir, file_id, 'images')
            os.makedirs(images_dir, exist_ok=True)
            masks_dir = os.path.join(label_out_dir, file_id, 'masks')
            os.makedirs(masks_dir, exist_ok=True)

            img = cv2.imread(os.path.join(root, fname), cv2.IMREAD_GRAYSCALE)
            try:
                labels = label(img, connectivity=2)
            except Exception as e:
                print(e)
            else:
                for i in range(1, labels.max() + 1):
                    label_img = (labels == i).astype(np.uint8) * 255

                    origin_fname = fname.replace('_Binary.tif', '.tiff')
                    copyfile(os.path.join(origin_dir, origin_fname), os.path.join(images_dir, origin_fname))

                    mask_fname = os.path.join(masks_dir, str(i)) + '.jpg'
                    cv2.imwrite(mask_fname, label_img)


if __name__ == '__main__':
    main()
