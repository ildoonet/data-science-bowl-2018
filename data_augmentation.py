import random
import cv2
import numpy as np


def resize_shortedge(data, target_size):
    img_h, img_w = data.img.shape[:2]
    scale = target_size / min(img_h, img_w)
    if img_h < img_w:
        new_h, new_w = target_size, round(scale * img_w)
    else:
        new_h, new_w = round(scale * img_h), target_size
    data.img = cv2.resize(data.img, (new_w, new_h))
    data.masks = [cv2.resize(mask, (new_w, new_h)) for mask in data.masks]
    return data


def random_crop_224(data):
    return random_crop(data, 224, 224)


def random_crop(data, w, h):
    """

    :param data: data.w >= w & data.h >= h
    :param w:
    :param h:
    :return: changed CellImageData
    """
    img_h, img_w = data.img.shape[:2]
    assert img_h >= h and img_w >= w, 'w=%d, h=%d' % (img_w, img_h)

    x = random.randint(0, img_w - w)
    y = random.randint(0, img_h - h)

    data.img = data.img[y:y+h, x:x+w, :]
    data.masks = [mask[y:y+h, x:x+w] for mask in data.masks]

    img_h2, img_w2 = data.img.shape[:2]
    assert img_h2 == h and img_w2 == w, 'w=%d->%d, h=%d->%d, target=(%d, %d)' % (img_w, img_w2, img_h, img_h2, w, h)

    return data


def data_to_segment_input(data):
    """
    :param data:
    :return: image(h, w, 1), mask(h, w, 1), masks(h, w, m)
    """
    return data[0].image(is_gray=True), data[0].single_mask(), data[0].multi_masks_batch()
