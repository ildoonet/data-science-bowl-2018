import random
import cv2
import numpy as np
from imgaug import augmenters as iaa
from scipy import ndimage
from scipy.ndimage import gaussian_filter, map_coordinates

from hyperparams import HyperParams


def erosion_mask(data):
    """
    As described in the original paper, Separation between cluttered cells is enhanced by using morphological algorithm.

    :param data: CellImageData
    :return: CellImageData
    """
    total_map = np.zeros_like(data.masks[0], dtype=np.uint8)
    masks = []
    for mask in data.masks:
        mask[total_map > 0] = 0
        mask = ndimage.morphology.binary_erosion(
            (mask > 0),
            border_value=1,
            iterations=HyperParams.get().pre_erosion_iter
        ).astype(np.uint8)
        total_map = total_map + mask

        masks.append(mask)
    data.masks = masks
    return data


def random_flip_lr(data):
    """
    randomly flip(50%) horizontally
    :param data: CellImageData
    :return: CellImageData
    """
    s = random.randint(0, 1)
    if s == 0:
        return data
    return flip(data, orientation=1)


def random_flip_ud(data):
    """
    randomly flip(50%) vertically
    :param data: CellImageData
    :return: CellImageData
    """
    s = random.randint(0, 1)
    if s == 0:
        return data
    return flip(data, orientation=0)


def flip(data, orientation=0):
    """
    flip CellImageData with the specified orientation(0=vertical, 1=horizontal)
    """
    # flip
    data.img = cv2.flip(data.img, orientation)
    data.masks = [cv2.flip(mask, orientation) for mask in data.masks]
    return data


def resize_shortedge_if_small(data, target_size):
    """
    resize the image ONLY IF its size is smaller than 'target_size'
    :param data: CellImageData
    :param target_size:
    :return:
    """
    img_h, img_w = data.img.shape[:2]
    if img_h < target_size or img_w < target_size:
        data = resize_shortedge(data, target_size)

    return data


def pad_if_small(data, target_size):
    img_h, img_w = data.img.shape[:2]
    padding = max(target_size - img_h, target_size - img_w)
    if padding <= 0:
        return data

    data.img = crop_mirror(data.img, 0, 0, img_w, img_h, padding)
    data.masks = [crop_mirror(mask, 0, 0, img_w, img_h, padding) for mask in data.masks]
    return data


def resize_shortedge(data, target_size):
    """
    Resize the image and masks as the shorter axis would be the same size of the target.
    :param data: CellImageData
    :return: CellImageData
    """
    img_h, img_w = data.img.shape[:2]
    scale = target_size / min(img_h, img_w)
    if img_h < img_w:
        new_h, new_w = target_size, round(scale * img_w)
    else:
        new_h, new_w = round(scale * img_h), target_size
    data.img = cv2.resize(data.img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    data.masks = [cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_AREA) for mask in data.masks]
    return data


def random_crop(data, w, h, padding=0):
    """
    Random-Crop cell image data(image, masks) with the specified size.
    :param data: CellImageData
    :return: CellImageData
    """
    img_h, img_w = data.img.shape[:2]

    x = random.randint(0, img_w - w)
    y = random.randint(0, img_h - h)

    crop(data, x, y, w, h, padding=padding)

    return data


def center_crop(data, w, h, padding=0):
    """
    Center-Crop cell image data(image, masks) with the specified size.
    :param data: CellImageData
    :return: CellImageData
    """
    img_h, img_w = data.img.shape[:2]

    x = (img_w - w) // 2
    y = (img_h - h) // 2

    crop(data, x, y, w, h, padding=padding)

    return data


def crop(data, x, y, w, h, padding=0):
    """
    Crop cell image data(image, masks) with the specified coordinate.
    :param data: CellImageData
    :return: CellImageData
    """
    assert x >= 0 and y >= 0 and w > 0 and h > 0

    img_h, img_w = data.img.shape[:2]
    assert img_h >= h and img_w >= w, 'w=%d, h=%d' % (img_w, img_h)

    data.img = crop_mirror(data.img, x, y, w, h, padding)
    data.masks = [mask[y:y + h, x:x + w] for mask in data.masks]

    img_h2, img_w2 = data.img.shape[:2]
    assert img_h2 == h+padding*2 and img_w2 == w+padding*2, 'w=%d->%d, h=%d->%d, target=(%d, %d) padding=%d' % (img_w, img_w2, img_h, img_h2, w, h, padding)

    return data


def crop_mirror(img, x, y, w, h, padding=0):
    assert x >= 0 and y >= 0 and w > 0 and h > 0

    if len(img.shape) == 3:
        padded_img = np.array([np.pad(ch, padding, 'reflect') for ch in img.transpose((2, 0, 1))]).transpose((1, 2, 0))
    else:
        padded_img = np.pad(img, padding, 'reflect')

    assert padded_img.shape[0] == img.shape[0] + padding * 2, (img.shape, padded_img.shape)
    assert padded_img.shape[1] == img.shape[1] + padding * 2, (img.shape, padded_img.shape)
    if len(img.shape) == 3:
        assert padded_img.shape[2] == img.shape[2], (img.shape, padded_img.shape)
    cropped_img = padded_img[y:y+h+padding*2, x:x+w+padding*2]
    return cropped_img


def random_scaling(data):
    """
    Randomly scale an image and masks.
    :param data: CellImageData
    :return: CellImageData
    """
    s = random.randint(0, 1)
    if s <= 0:
        return data
    img_h, img_w = data.img.shape[:2]
    scale_f1 = HyperParams.get().pre_scale_f1
    scale_f2 = HyperParams.get().pre_scale_f2
    new_w = int(random.uniform(1.-scale_f1, 1.+scale_f2) * img_w)
    new_h = int(random.uniform(1.-scale_f1, 1.+scale_f2) * img_h)

    data.img = cv2.resize(data.img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    data.masks = [cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_AREA) for mask in data.masks]
    data.img_w, data.img_h = new_w, new_h
    return data


def random_affine(data):
    """
    Randomly apply affine transformations including rotation, shearing, translation.
    :param data: CellImageData
    :return: CellImageData
    """
    s = random.randint(0, 2)
    if s >= 1:
        return data
    rand_rotate = np.random.randint(-HyperParams.get().pre_affine_rotate, HyperParams.get().pre_affine_rotate)
    rand_shear = np.random.randint(-HyperParams.get().pre_affine_shear, HyperParams.get().pre_affine_shear)
    rand_translate = np.random.uniform(-HyperParams.get().pre_affine_translate, HyperParams.get().pre_affine_translate)

    aug = iaa.Affine(scale=1.0, translate_percent=rand_translate, rotate=rand_rotate, shear=rand_shear, cval=0, mode='reflect')
    data.img = aug.augment_image(data.img)
    data.masks = [aug.augment_image(mask) for mask in data.masks]
    return data


def random_color(data):
    """
    Changing Color Randomly for Augmentation
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
    """
    Invert / Contrast Normalization / Hue&Saturation Augmentation.
    Currently, this is not used since it degrade the performance.
    """
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
    :return: image(h, w, 1), mask(h, w, 1), masks(h, w, m), (optional) unet weights for mask
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
    """
    CellImageData to numpy Images
    :param data: CellImageData
    :return: List of Image, data index, data size, (optional) unet weights for mask
    """
    vals = [
        data[0].image(is_gray=is_gray),
        np.array([data[0].target_id], dtype=np.object),
        np.array([data[0].img_h, data[0].img_w], dtype=np.int32),
    ]
    if unet_weight:
        vals.append(data[0].unet_weights())
    return vals


def data_to_normalize01(data):
    """
    Normalize images to have values between 0.0 and 1.0.
    :param data: numpy array or CellImageData
    :return: numpy array or CellImageData
    """
    if isinstance(data, np.ndarray):
        return data.astype(np.float32) / 255
    data.img = data.img.astype(np.float32) / 255
    return data


def data_to_normalize1(data):
    """
    Normalize images to have values between -1.0 and 1.0.
    :param data: numpy array or CellImageData
    :return: numpy array or CellImageData
    """
    if isinstance(data, np.ndarray):
        return data.astype(np.float32) / 128 - 1.0
    data.img = data.img.astype(np.float32) / 128 - 1.0
    return data


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
    masks = [mask[:, :, 0] for mask in masks]

    return image, masks


def mask_size_normalize(data, target_size=None):
    s = random.randint(0, 1)
    if s <= 0 and target_size is None:
        data = random_scaling(data)
        return data

    # getting maximum size of masks
    maximum_size = get_max_size_of_masks(data.masks)
    if maximum_size <= 1:
        return data

    # normalize by the target size
    if target_size is None:
        target_size = random.uniform(HyperParams.get().pre_size_norm_min, HyperParams.get().pre_size_norm_max)
    shorter_edge_size = min(data.img.shape[:2])
    size_factor = target_size / maximum_size
    size_factor = min(5000 / shorter_edge_size, size_factor)
    size_factor = max(120 / shorter_edge_size, size_factor)

    target_edge_size = int(shorter_edge_size * size_factor)

    data = resize_shortedge(data, target_edge_size)

    return data


def get_max_size_of_masks(masks):
    def _bbox(img):
        rows = np.any(img, axis=1)
        cols = np.any(img, axis=0)
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]

        return rmin, rmax, cmin, cmax

    maximum_size = 1  # in pixel
    for mask in masks:
        rmin, rmax, cmin, cmax = _bbox(mask)
        maximum_size = max(maximum_size, rmax - rmin)
        maximum_size = max(maximum_size, cmax - cmin)
    return maximum_size

# TODO : Image Drop Augmentation, imgaug pepper? dropout? blur? constrast?


# TODO : Thick line Occlusion, add/multiply some values? sharpen?
def random_add_thick_area(data):
    s = random.randint(0, 9)
    if 0 < s:
        return data

    img = data.img
    overlay = img.copy()

    pt1, pt2 = np.random.randint(0, max(img.shape[0], img.shape[1]), size=(2, 2))
    cv2.rectangle(overlay, tuple(pt1), tuple(pt2), (255, 255, 255), thickness=cv2.FILLED)

    alpha = np.random.random()
    return cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

