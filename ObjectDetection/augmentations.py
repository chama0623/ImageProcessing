# library of preprocessing and augmentation

from random import uniform
import cv2
import numpy as np
from numpy import random

class Compose(object): # objectはextendsを明記しなくてもよい
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
            
class RandomContrast(object):
    """change contrast randomly

    Args:
        object (_type_): _description_
    """
    
    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."
        
    def __call__(self, image, boxes=None, labels=None):
        if random.randint(2):
            alpha = random.uniform(self.lower, self.upper)
            image *= alpha
            
        return image, boxes, labels
    
class ConvertColor(object):
    """transform GBR to HSV or HSV to GBR

    Args:
        object (_type_): _description_
    """
    
    def __init__(self, current="BGR", transform="HSV"):
        self.current = current
        self.transform = transform
        
    def __call__(self, image, boxes=None, labels=None):
        if self.current == "BGR" and self.transform == "HSV":
            image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        if self.current == "HSV" and self.transform == "BGR":
            image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
        else:
            raise NotImplementedError # 実装されていないことを表すエラー
        
        return image, boxes, labels
    
class RandomSaturation(object):
    """change saturation randomly

    Args:
        object (_type_): _description_
    """
    
    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."
        
    def __call__(self, image, boxes=None, labels=None):
        if random.randint(2):
            image[:,:,1] *= random.uniform(self.lower, self.upper)
            
        return image, boxes, labels
    
class RandomHue(object):
    """change hue randomly

    Args:
        object (_type_): _description_
    """
    
    def __init__(self, delta=18.0):
        assert delta >= 0.0 and delta <=360.0
        self.delta = delta
        
        def __call__(self, image, boxes=None, labels=None):
            if random.randint(2):
                image[:,:,0] += random,uniform(-self.delta, self.delta)
                image[:,:,0][image[:,:,0] > 360.0] -=360.0
                image[:,:,0][image[:,:,0] < 0.0] +=360.0
            
            return image, boxes, labels
        
class RandomLightingNoise(object):
    """add lighting noise randomly

    Args:
        object (_type_): _description_
    """
    def __init__(self):
        self.perms = ((0, 1, 2), (0, 2, 1),
                      (1, 0, 2), (1, 2, 0),
                      (2, 0, 1), (2, 1, 0))
        
    def __call__(self, image, boxes=None, labels=None):
        if random.randint(2):
            swap = self.perms[random.randint(len(self.perms))]
            shuffle = SwapChannels(swap)
            image = shuffle(image)
        
        return image, boxes, labels       

class SwapChannels(object):
    def __init__(self, swaps):
        """swap channels

        Args:
            swaps (int triple): final order of channels
            eg: (2, 1, 0)
        """
        self.swaps = swaps
        
    def __call__(self, image):
        image = image[:, :, self.swaps]
        return image
    
class PhotometricDistort(object):
    """change contrast, saturation and hue, and add light noise

    Args:
        object (_type_): _description_
    """
    def __init__(self):
        self.pd = [
            RandomContrast(),
            ConvertColor(transform="HSV"),
            RandomSaturation(),
            RandomHue(),
            ConvertColor(current="HSV", transform="BGR"),
            RandomContrast()
        ]
        self.rand_brightness = RandomBrightness()
        self.rand_light_noise = RandomLightingNoise()
        
    def __call__(self, image, boxes, labels):
        im = image.copy()
        im, boxes, labels = self.rand_brightness(im, boxes, labels)
        if random.randint(2):
            distort = Compose(self.pd[:-1]) # contrast saturation, hue
        else:
            distort = Compose(self.pd[1:]) # saturation, hue, contrast
        
        im, boxes, labels = distort(im, boxes, labels)
        return self.rand_light_noise(im, boxes, labels)
    
class Expand(object):
    """expand image randomly

    Args:
        object (_type_): _description_
    """
    
    def __init__(self, mean):
        self.mean = mean
        
    def __call__(self, image, boxes, labels):
        if random.randint(2):
            return image, boxes, labels
        
        height, width, depth = image.shape
        ratio = random.uniform(1, 4)
        left = random.uniform(0, width*ratio -width)
        top = random.uniform(0, height*ratio -height)
        
        expand_image = np.zeros(
            (int(height*ratio), int(width*ratio), depth),
            dtype = image.dtype
        )
        expand_image[:, :, :] = self.mean
        expand_image[int(top):int(top+height), int(left):int(left+width)] = image
        image = expand_image
        
        # 拡大した画像に合わせてB-Boxを加工
        boxes = boxes.copy()
        boxes[:, :2] += (int(left), int(top))
        boxes[:, 2:] += (int(left), int(top))
        
        return image, boxes, labels
    
class RandomMirror(object):
    """mirroring image randomly

    Args:
        object (_type_): _description_
    """
    def __call__(self, image, boxes, labels):
        _, width, _ = image.shape
        if random.randint(2):
            image = image[:, ::-1]
            boxes = boxes.copy()
            boxes[:, 0::2] = width - boxes[:, 2::-2]
        
        return image, boxes, labels
    
class ToPercentCoords(object):
    """normaliza annotation data to 0.0~1.0

    Args:
        object (_type_): _description_
    """
    
    def __call__(self, image, boxes=None, labels=None):
        height, width, channels = image.shape
        boxes[:, 0] /= width
        boxes[:, 2] /= width
        boxes[:, 1] /= height
        boxes[:, 3] /= height
        
        return image, boxes, labels
    
class Resize(object):
    """resize image (default: 300x300)

    Args:
        object (_type_): _description_
    """
    def __init__(self, size=300):
        self.size = size
        
    def __call__(self, image, boxes=None, labels=None):
        image = cv2.resize(image, (self.size, self.size))
        return image, boxes, labels
    
class SubtractMeans(object):
    """subtract mean from RGB value

    Args:
        object (_type_): _description_
    """
    def __init__(self, mean):
        self.mean = np.array(mean, dtype=np.float32)
        
    def __call__(self, image, boxes=None, labels=None):
        image = image.astype(np.float32)
        image -= self.mean
        return image, boxes, labels