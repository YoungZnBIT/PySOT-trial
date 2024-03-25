

import numpy as np

import cv2
from glob import glob
from os.path import join
import platform

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

SYSTEM = platform.system()

RGB_MEAN = np.reshape(np.array([0.485, 0.456, 0.406]), (1, 1, 1, 3))
RGB_STD = np.reshape(np.array([0.229, 0.224, 0.225]), (1, 1, 1, 3))


def get_frames(video_name):
    """
    Read image frame by Opencv

    :param video_name: str, image or video file path
    :return: image matrix
    """
    if not video_name:
        cap = cv2.VideoCapture(0)
        # warmup
        for i in range(5):
            cap.read()
        while True:
            ret, frame = cap.read()
            if ret:
                yield frame
            else:
                break
    elif video_name.endswith('avi') or video_name.endswith('mp4'):
        cap = cv2.VideoCapture(video_name)
        while True:
            ret, frame = cap.read()
            if ret:
                yield frame
            else:
                break
    else:
        images = glob(join(video_name, '*.jp*'))

        if SYSTEM == 'Linux':
            images = sorted(images, key=lambda x: int(x.split('/')[-1].split('.')[0]))

        for img in images:
            frame = cv2.imread(img)
            yield frame


def crop_roi(image, input_size, crop_bbox, padding=(0, 0, 0)):
    """
    crop and resize image like SiamRPN++: 使用仿射变换

    :param image:
    :param input_size:
    :param crop_bbox:
    :param padding:
    :return:
    """
    crop_bbox = [float(x) for x in crop_bbox]
    a = (input_size - 1) / (crop_bbox[2] - crop_bbox[0])
    b = (input_size - 1) / (crop_bbox[3] - crop_bbox[1])
    c = -a * crop_bbox[0]
    d = -b * crop_bbox[1]
    mapping = np.array([[a, 0, c],
                        [0, b, d]]).astype(np.float)
    crop = cv2.warpAffine(image, mapping, (input_size, input_size), borderMode=cv2.BORDER_CONSTANT, borderValue=padding)
    return crop


def letterbox_image(image, input_size, bbox=None, padding=None):
    """
    resize image like YOLO: unchanging aspect ratio by padding

    :param image:
    :param input_size:
    :param bbox:
    :param padding:
    :return:
    """
    cv2.setNumThreads(0)
    cv2.ocl.setUseOpenCL(False)

    if bbox is None:
        bbox = []
    if padding is None:
        padding = np.array([128, 128, 128])[None, None, :]
    ih, iw = image.shape[:2]

    if not isinstance(input_size, list) and not isinstance(input_size, list):
        input_size = (input_size, input_size)

    scale = min(input_size[1] / iw, input_size[0] / ih)
    nw = int(iw * scale)
    nh = int(ih * scale)
    dx = (input_size[1] - nw) // 2
    dy = (input_size[0] - nh) // 2

    # image = cv2.resize(image, (nw, nh))
    try:
        image = cv2.resize(image, (nw, nh))
    except:
        print('image shape: ')
        print(image.shape)
        print('iw: {:d}, ih: {:d}, nw: {:d}, nh: {:d}, scale: {:.4f}'.format(iw, ih, nw, nh, scale))
        print(image)

    new_image = padding * np.ones(shape=(input_size[1], input_size[0], 3))
    new_image = new_image.astype(np.uint8)
    new_image[dy:(dy+nh), dx:(dx+nw), :] = image
    offset = np.array([dx, dy, dx, dy], dtype=np.float32)
    if len(bbox):
        bbox = bbox * scale + offset
        return new_image, bbox
    else:
        return new_image


def rgb_normalize(bgr_images, mobilenet=False):
    bgr_images = np.maximum(np.minimum(bgr_images, 255.0), 0.0)
    rgb_images = bgr_images[..., ::-1]
    if not mobilenet:
        rgb_images = rgb_images / 255.
        rgb_images = (rgb_images - RGB_MEAN) / RGB_STD
    else:
        rgb_images = rgb_images / 127.5
        rgb_images = rgb_images - 1.
    return rgb_images.astype(np.float32)


def normalize_batch_cuda(tensor, mean, std, inplace=False):
    """Normalize a tensor image with mean and standard deviation.

    .. note::
        This transform acts out of place by default, i.e., it does not mutates the input tensor.

    See :class:`~torchvision.transforms.Normalize` for more details.

    Args:
        tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.
        inplace(bool,optional): Bool to make this operation inplace.

    Returns:
        Tensor: Normalized Tensor image.
    """
    if not inplace:
        tensor = tensor.clone()
    tensor = tensor.mul(1.0 / 255.0).clamp(0.0, 1.0)
    tensor.sub_(mean).div_(std)
    return tensor


def get_subwindow(im, pos, model_sz, original_sz, avg_chans):
    """
    :param im: bgr based image
    :param pos: center position
    :param model_sz: exemplar size
    :param original_sz: original size
    :param avg_chans: channel average
    :param mobilenet:
    :return:
    """
    if isinstance(pos, float):
        pos = [pos, pos]
    sz = original_sz
    im_sz = im.shape
    c = (original_sz + 1.) / 2.

    # crop patch 在原图像上的位置
    context_xmin_ = np.floor(pos[0] - c + 0.5)
    context_xmax_ = context_xmin_ + sz - 1.
    context_ymin_ = np.floor(pos[1] - c + 0.5)
    context_ymax_ = context_ymin_ + sz - 1.

    # 原图像在 padding 图像上的位置
    left_pad = int(max(0., -context_xmin_))
    top_pad = int(max(0., -context_ymin_))
    right_pad = int(max(0., context_xmax_ - im_sz[1] + 1.))
    bottom_pad = int(max(0., context_ymax_ - im_sz[0] + 1.))

    # crop patch 在 padding 图像上的位置
    context_xmin = context_xmin_ + left_pad
    context_xmax = context_xmax_ + left_pad
    context_ymin = context_ymin_ + top_pad
    context_ymax = context_ymax_ + top_pad

    r, c, k = im.shape
    if any([top_pad, bottom_pad, left_pad, right_pad]):
        size = (r + top_pad + bottom_pad, c + left_pad + right_pad, k)
        te_im = np.zeros(size, np.uint8)
        te_im[top_pad:top_pad + r, left_pad:left_pad + c, :] = im
        if top_pad:
            te_im[0:top_pad, left_pad:left_pad + c, :] = avg_chans
        if bottom_pad:
            te_im[r + top_pad:, left_pad:left_pad + c, :] = avg_chans
        if left_pad:
            te_im[:, 0:left_pad, :] = avg_chans
        if right_pad:
            te_im[:, c + left_pad:, :] = avg_chans
        im_patch = te_im[int(context_ymin):int(context_ymax + 1), int(context_xmin):int(context_xmax + 1), :]
    else:
        im_patch = im[int(context_ymin):int(context_ymax + 1), int(context_xmin):int(context_xmax + 1), :]

    if not np.array_equal(model_sz, original_sz):
        im_patch = cv2.resize(im_patch, (model_sz, model_sz), interpolation=cv2.INTER_CUBIC)
    im_patch = im_patch[np.newaxis, :, :, :]
    im_patch = im_patch.astype(np.float32)
    return im_patch
