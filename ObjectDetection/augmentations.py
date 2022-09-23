import cv2
import numpy as np
from numpy import random

class Compose(object):
    def __init__(self, transforms):
        """

        Args:
            transforms (List[Transforms]): list of transform
        
        Examples:
        >>> augmentations.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
        """
        self.transforms = transforms
        
        def __call__(self, img, boxes=None, labels=None):
            for t in self.transforms:
                img, boxes, labels = t(img, boxes, labels)
            return img, boxes, labels
        
class ConvertFromInts(object):
    """transform int pixel data to float32

    Args:
        object (_type_): _description_
    """
    def __call__(self, image, boxes=None, labels=None):
        return image.astype(np.float32), boxes, labels
    
class ToAbsoluteCoords(object):
    """undo normalization of annotation data

    Args:
        object (_type_): _description_
    """
    
    def __call__(self, image, boxes=None, labels=None):
        height, width, channels = image.shape
        boxes[:,0] *=width
        boxes[:,2] *=width
        boxes[:,1] *=height
        boxes[:,3] *=height
        
        return image, boxes, labels
    
class RandomBrightness(object):
    """change brightness randomly

    Args:
        object (_type_): _description_
    """
    
    def __init__(self, delta=32):
        assert delta >= 0.0
        assert delta <=255.0
        self.delta = delta
        
    def __call__(self, image ,boxes=None, labels=None):
        if random.randint(2): # 30%の確率で
            delta = random.uniform(-self.delta, self.delta)
            image += delta
            
        return image, boxes, labels
            