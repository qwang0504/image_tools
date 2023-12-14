from typing import Optional
from .convert_gpu import im2single_GPU
import cupy as cp
from cupy.typing import NDArray as CuNDArray
from cupyx.scipy import ndimage as cu_ndi

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