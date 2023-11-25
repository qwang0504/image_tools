from numpy.typing import NDArray
import numpy as np
from typing import Optional
from scipy import ndimage
import cv2
from .convert import im2single

def enhance(
        image: NDArray, 
        contrast: float = 1.0,
        gamma: float = 1.0,
        brightness: float = 0.0, 
        blur_size_px: Optional[int] = None, 
        medfilt_size_px: Optional[int] = None
    ) -> NDArray:

    if gamma <= 0:
        raise ValueError('gamma should be > 0')
    if not (-1 <= brightness <= 1):
        raise ValueError('brightness should be between -1 and 1')
    
    # make sure image is single precision
    output = im2single(image)

    # brightness, contrast, gamma
    output = contrast*(output+brightness)**gamma
    
    # clip between 0 and 1
    np.clip(output, 0, 1, out=output)

    # blur
    if (blur_size_px is not None) and (blur_size_px > 0):
        output = cv2.boxFilter(output, -1, (blur_size_px, blur_size_px))

    # median filter
    if (medfilt_size_px is not None) and (medfilt_size_px > 0):
        ndimage.median_filter(output, size = medfilt_size_px, output=output)

    return output