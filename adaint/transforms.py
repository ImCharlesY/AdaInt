import random
import numpy as np
from PIL import Image
from torch.nn.modules.utils import _pair
from torchvision.transforms import ColorJitter

from mmedit.datasets.registry import PIPELINES


@PIPELINES.register_module()
class RandomRatioCrop(object):
    r"""Random crop the image.

    Args:
        keys (Sequence[str]): The images to be cropped.
        crop_ratio (tuple): Expected ratio of size after cropping.
        isotropic (bool, optional): Whether to perform isotropic cropping.
            Default: False.
    """

    def __init__(self, keys, crop_ratio, isotropic=False):
        self.crop_ratio = _pair(crop_ratio)
        self.isotropic = isotropic
        self.keys = keys

    def _get_cropbox(self, img):
        ratio_h = random.uniform(*self.crop_ratio)
        ratio_w = ratio_h if self.isotropic else random.uniform(*self.crop_ratio)
        crop_size = (int(img.shape[0] * ratio_h), int(img.shape[1] * ratio_w))
        margin_h = max(img.shape[0] - crop_size[0], 0)
        margin_w = max(img.shape[1] - crop_size[1], 0)
        offset_h = random.randint(0, margin_h)
        offset_w = random.randint(0, margin_w)
        crop_y1, crop_y2 = offset_h, offset_h + crop_size[0]
        crop_x1, crop_x2 = offset_w, offset_w + crop_size[1]
        return crop_y1, crop_y2, crop_x1, crop_x2
    
    def __call__(self, results):
        y1, y2, x1, x2 = self._get_cropbox(results[self.keys[0]])
        for key in self.keys:
            results[key] = results[key][y1:y2, x1:x2, :]
            results[f'{key}_crop_size'] = results[key].shape[:2]
        return results

    def __repr__(self):
        return self.__class__.__name__ + '(crop_ratio={}, isotropic={}, keys={})'.format(
            self.crop_ratio, self.isotropic, self.keys)
        
        
@PIPELINES.register_module()
class FlexibleRescaleToZeroOne(object):
    r"""Transform the images into a range between 0 and 1.
    
    Compared to the original `RescaleToZeroOne` provided by mmedit, this implementation
    supports 16-bit images.

    Args:
        keys (Sequence[str]): The images to be rescaled.
        precision (int, optional): precision of the float type. Default: 32.
    """

    def __init__(self, keys, precision=32):
        assert precision in [16, 32, 64]
        self.precision = 'float{}'.format(precision)
        self.keys = keys

    def _to_float(self, img):
        if img.dtype == np.uint8:
            factor = 255
        elif img.dtype == np.uint16:
            factor = 65535
        else:
            factor = 1
        img = img.astype(self.precision) / factor
        return img.clip(0, 1)

    def __call__(self, results):
        for key in self.keys:
            results[key] = self._to_float(results[key])
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += '(precision={}, keys={})'.format(
            self.precision, self.keys)
        return repr_str
    


@PIPELINES.register_module
class RandomColorJitter(object):
    r"""(This is a cvframe interface to the torchvision.ColorJitter)
    Randomly change the brightness, contrast, saturation and hue of an image.
    If the image is torch Tensor, it is expected
    to have [..., 3, H, W] shape, where ... means an arbitrary number of leading dimensions.
    If img is PIL Image, mode "1", "L", "I", "F" and modes with transparency (alpha channel) are not supported.

    Args:
        keys (Sequence[str]): The images to be applied jitter.
        brightness (float or tuple of float (min, max), optional): How much to jitter brightness.
            brightness_factor is chosen uniformly from [max(0, 1 - brightness), 1 + brightness]
            or the given [min, max]. Should be non negative numbers.
        contrast (float or tuple of float (min, max), optional): How much to jitter contrast.
            contrast_factor is chosen uniformly from [max(0, 1 - contrast), 1 + contrast]
            or the given [min, max]. Should be non negative numbers.
        saturation (float or tuple of float (min, max), optional): How much to jitter saturation.
            saturation_factor is chosen uniformly from [max(0, 1 - saturation), 1 + saturation]
            or the given [min, max]. Should be non negative numbers.
        hue (float or tuple of float (min, max), optional): How much to jitter hue.
            hue_factor is chosen uniformly from [-hue, hue] or the given [min, max].
            Should have 0<= hue <= 0.5 or -0.5 <= min <= max <= 0.5.
    """

    def __init__(self, keys, brightness=0, contrast=0, saturation=0, hue=0):
        self.keys = keys
        self._transform = ColorJitter(brightness, contrast, saturation, hue)

    def transform(self, img):
        return np.array(self._transform(Image.fromarray(img)))

    def __call__(self, results):
        for key in self.keys:
            results[key] = self.transform(results[key])
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += '(brightness={}, contrast={}, saturation={}, hue={}, keys={})'.format(
            self._transform.brightness, self._transform.contrast,
            self._transform.saturation, self._transform.hue, self.keys)
        


@PIPELINES.register_module()
class FlipChannels(object):
    r"""Flip the color channels.

    Args:
        keys (Sequence[str]): The images to be flipped.
    """

    def __init__(self, keys):
        self.keys = keys

    def __call__(self, results):
        for key in self.keys:
            results[key] = results[key][..., ::-1]
        return results

    def __repr__(self):
        return self.__class__.__name__ + '(keys={})'.format(self.keys)