from numpy.typing import NDArray
import numpy as np
from typing import Optional
from scipy.signal import medfilt2d
import cv2
from .convert import im2single, im2single_GPU

import cupy as cp
from cupy.typing import NDArray as CuNDArray
from cupyx.scipy import ndimage as cu_ndi

# TODO this appears to be mostly single-threaded, profile it
# NOTE median filter becomes prohibitively slow as kernel size increases

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
    output = contrast*(output**gamma)+brightness
    
    # clip between 0 and 1
    np.clip(output, 0, 1, out=output)

    # blur
    if (blur_size_px is not None) and (blur_size_px > 0):
        output = cv2.boxFilter(output, -1, (blur_size_px, blur_size_px))

    # median filter
    if (medfilt_size_px is not None) and (medfilt_size_px > 0):
        medfilt_size_px = medfilt_size_px + int(medfilt_size_px % 2 == 0)
        output = medfilt2d(output, kernel_size = medfilt_size_px)

    return output

def enhance_GPU(
        image: CuNDArray, 
        contrast: float = 1.0,
        gamma: float = 1.0,
        brightness: float = 0.0, 
        blur_size_px: Optional[int] = None, 
        medfilt_size_px: Optional[int] = None
    ) -> CuNDArray:

    if gamma <= 0:
        raise ValueError('gamma should be > 0')
    if not (-1 <= brightness <= 1):
        raise ValueError('brightness should be between -1 and 1')
    
    # make sure image is single precision
    output = im2single_GPU(image)

    # brightness, contrast, gamma
    output = contrast*(output**gamma)+brightness
    
    # clip between 0 and 1
    cp.clip(output, 0, 1, out=output)

    # blur
    if (blur_size_px is not None) and (blur_size_px > 0):
        cu_ndi.gaussian_filter(output, blur_size_px, output=output)

    # median filter
    if (medfilt_size_px is not None) and (medfilt_size_px > 0):
        cu_ndi.median_filter(output, medfilt_size_px, output=output)

    return output