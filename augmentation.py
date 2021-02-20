import numpy as np
from torchvision import transforms

class ConvertFromInts(object):
    def __call__(self, sample):
        sample['image'] = np.divide(sample['image'].astype(np.float32), 255.0)
        return sample

class BaseTransform(object):
    def __init__(self):
        self.augment = transforms.Compose([
            ConvertFromInts(),
        ])

    def __call__(self, sample):
        return self.augment(sample)