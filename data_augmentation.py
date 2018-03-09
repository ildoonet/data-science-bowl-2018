import random
import cv2
import numpy as np
from imgaug import augmenters as iaa
from scipy import ndimage
from scipy.ndimage import gaussian_filter, map_coordinates


def erosion_mask(data):
    total_map = np.zeros_like(data.masks[0], dtype=np.uint8)
    masks = []
    for mask in data.masks:
        mask[total_map > 0] = 0
        mask = ndimage.morphology.binary_erosion((mask > 0), border_value=1).astype(np.uint8)
        total_map = total_map + mask

        masks.append(mask)
    data.masks = masks
    return data


def random_flip_lr(data):
    s = random.randint(0, 1)
    if s == 0:
        return data
    return flip(data, orientation=0)


def random_flip_ud(data):
    s = random.randint(0, 1)
    if s == 0:
        return data
    return flip(data, orientation=1)


def flip(data, orientation=0):
    # flip horizontally
    data.img = cv2.flip(data.img, orientation)
    data.masks = [cv2.flip(mask, orientation) for mask in data.masks]
    return data


def resize_shortedge_if_small_224(data):
    return resize_shortedge_if_small(data, 224)


def resize_shortedge_if_small(data, target_size):
    """
    resize the image ONLY IF its size is smaller than 'target_size'
    """
    img_h, img_w = data.img.shape[:2]
    if img_h < target_size or img_w < target_size:
        data = resize_shortedge(data, target_size)
    return data


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
    img_h, img_w = data.img.shape[:2]

    x = random.randint(0, img_w - w)
    y = random.randint(0, img_h - h)

    crop(data, x, y, w, h)

    return data


def center_crop_224(data):
    return center_crop(data, 224, 224)


def center_crop(data, w, h):
    img_h, img_w = data.img.shape[:2]

    x = (img_w - w) // 2
    y = (img_h - h) // 2

    crop(data, x, y, w, h)

    return data


def crop(data, x, y, w, h):
    assert x >= 0 and y >= 0 and w > 0 and h > 0

    img_h, img_w = data.img.shape[:2]
    assert img_h >= h and img_w >= w, 'w=%d, h=%d' % (img_w, img_h)

    data.img = data.img[y:y + h, x:x + w, :]
    data.masks = [mask[y:y + h, x:x + w] for mask in data.masks]

    img_h2, img_w2 = data.img.shape[:2]
    assert img_h2 == h and img_w2 == w, 'w=%d->%d, h=%d->%d, target=(%d, %d)' % (img_w, img_w2, img_h, img_h2, w, h)

    return data


def random_scaling(data):
    s = random.randint(0, 1)
    if s == 0:
        return data
    img_h, img_w = data.img.shape[:2]
    scale_f1 = 0.4
    scale_f2 = 0.4
    new_w = int(random.uniform(1.-scale_f1, 1.+scale_f2) * img_w)
    new_h = int(random.uniform(1.-scale_f1, 1.+scale_f2) * img_h)

    data.img = cv2.resize(data.img, (new_w, new_h))
    data.masks = [cv2.resize(mask, (new_w, new_h)) for mask in data.masks]
    data.img_w, data.img_h = new_w, new_h
    return data


def random_affine(data):
    s = random.randint(0, 1)
    if s <= 0:
        return data
    rand_rotate = np.random.randint(-45, 45)
    rand_shear = np.random.randint(-10, 10)
    rand_translate = np.random.uniform(-0.1, 0.1)

    aug = iaa.Affine(scale=1.0, translate_percent=rand_translate, rotate=rand_rotate, shear=rand_shear, cval=0, mode='reflect')
    data.img = aug.augment_image(data.img)
    data.masks = [aug.augment_image(mask) for mask in data.masks]
    return data


def random_color(data):
    """
    reference : https://github.com/neptune-ml/data-science-bowl-2018/blob/master/augmentation.py
    """
    s = random.randint(0, 1)
    if s <= 0:
        return data
    aug = iaa.Sequential([
        # Color
        iaa.OneOf([
            iaa.Sequential([
                iaa.ChangeColorspace(from_colorspace="RGB", to_colorspace="HSV"),
                iaa.WithChannels(0, iaa.Add((0, 100))),
                iaa.ChangeColorspace(from_colorspace="HSV", to_colorspace="RGB")]),
            iaa.Sequential([
                iaa.ChangeColorspace(from_colorspace="RGB", to_colorspace="HSV"),
                iaa.WithChannels(1, iaa.Add((0, 100))),
                iaa.ChangeColorspace(from_colorspace="HSV", to_colorspace="RGB")]),
            iaa.Sequential([
                iaa.ChangeColorspace(from_colorspace="RGB", to_colorspace="HSV"),
                iaa.WithChannels(2, iaa.Add((0, 100))),
                iaa.ChangeColorspace(from_colorspace="HSV", to_colorspace="RGB")]),
            iaa.WithChannels(0, iaa.Add((0, 100))),
            iaa.WithChannels(1, iaa.Add((0, 100))),
            iaa.WithChannels(2, iaa.Add((0, 100)))
        ])
    ], random_order=True)
    data.img = aug.augment_image(data.img)
    # data.masks = [aug.augment_image(mask) for mask in data.masks]
    return data


def random_color2(data):
    aug = iaa.Sequential([
        iaa.Invert(0.25, per_channel=False),
        iaa.ContrastNormalization((0.7, 1.4)),
        iaa.Sometimes(0.2, iaa.AddToHueAndSaturation((-45, 45), per_channel=True))
    ])
    data.img = aug.augment_image(data.img)
    return data


def data_to_segment_input(data, is_gray=True, unet_weight=False):
    """
    :param data: CellImageData
    :return: image(h, w, 1), mask(h, w, 1), masks(h, w, m)
    """
    vals = [
        data[0].image(is_gray=is_gray),
        data[0].single_mask(),
        data[0].multi_masks_batch()
    ]
    if unet_weight:
        vals.append(data[0].unet_weights())
    return vals


def data_to_image(data, is_gray=True, unet_weight=False):
    vals = [
        data[0].image(is_gray=is_gray),
        np.array([data[0].target_id], dtype=np.object),
        np.array([data[0].img_h, data[0].img_w], dtype=np.int32),
    ]
    if unet_weight:
        vals.append(data[0].unet_weights())
    return vals


def data_to_normalize01(data):
    return data.astype(np.float32) / 255


def data_to_normalize1(data):
    return data.astype(np.float32) / 128 - 1.0


def data_to_elastic_transform_wrapper(data):
    i, ms = data_to_elastic_transform(data, data.img.shape[1] * 2, data.img.shape[1] * 0.08, data.img.shape[1] * 0.08)
    data.img = i
    data.masks = ms
    return data


# Function to distort image using elastic transformation technique
def data_to_elastic_transform(data, alpha, sigma, alpha_affine, random_state=None):
    """Elastic deformation of images as described in [Simard2003]_ (with modifications).
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
         Convolutional Neural Networks applied to Visual Document Analysis", in
         Proc. of the International Conference on Document Analysis and
         Recognition, 2003.

     Based on https://gist.github.com/erniejunior/601cdf56d2b424757de5
    """
    if random_state is None:
        random_state = np.random.RandomState(None)

    shape = data.img.shape
    shape_size = shape[:2]

    # Define random affine matrix
    center_square = np.float32(shape_size) // 2
    square_size = min(shape_size) // 3
    pts1 = np.float32([center_square + square_size, [center_square[0] + square_size, center_square[1] - square_size],
                       center_square - square_size])
    pts2 = pts1 + random_state.uniform(-alpha_affine, alpha_affine, size=pts1.shape).astype(np.float32)
    M = cv2.getAffineTransform(pts1, pts2)

    # Apply M to image
    image = cv2.warpAffine(data.img, M, shape_size[::-1], borderMode=cv2.BORDER_REFLECT_101)
    masks = [cv2.warpAffine(mask, M, shape_size[::-1], borderMode=cv2.BORDER_REFLECT_101) for mask in data.masks]

    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    dz = np.zeros_like(dx)

    x, y, z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))
    indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1)), np.reshape(z, (-1, 1))

    masks = [np.expand_dims(mask, axis=-1) for mask in masks]

    image = map_coordinates(image, indices, order=1, mode='reflect').reshape(shape)
    masks = [map_coordinates(mask, indices, order=1, mode='reflect').reshape(shape) for mask in masks]
    masks = [mask[:,:,0] for mask in masks]

    return image, masks
