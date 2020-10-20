import random

from PIL import Image


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, mask, depth):
        assert img.size == mask.size
        assert img.size == depth.size
        for t in self.transforms:
            img, mask, depth = t(img, mask, depth)
        return img, mask, depth

class Resize(object):
    def __init__(self, size):
        self.size = tuple(reversed(size))  # size: (h, w)

    def __call__(self, img, mask, depth):
        assert img.size == mask.size
        assert img.size == depth.size
        return img.resize(self.size, Image.BILINEAR), mask.resize(self.size, Image.BILINEAR), depth.resize(self.size, Image.BILINEAR)


class RandomCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img, mask, depth):
        assert img.size == mask.size
        assert img.size == depth.size
        w, h = img.size

        x1 = random.randint(0, w - self.size)
        y1 = random.randint(0, h - self.size)
        return img.crop((x1, y1, x1 + self.size, y1 + self.size)), mask.crop((x1, y1, x1 + self.size, y1 + self.size)), depth.crop((x1, y1, x1 + self.size, y1 + self.size))


class RandomHorizontallyFlip(object):
    def __call__(self, img, mask, depth):
        if random.random() < 0.5:
            return img.transpose(Image.FLIP_LEFT_RIGHT), mask.transpose(Image.FLIP_LEFT_RIGHT), depth.transpose(Image.FLIP_LEFT_RIGHT)
        return img, mask, depth
