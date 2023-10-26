from numpy.typing import NDArray
from typing import Optional
from scipy import ndimage
import cv2

#TODO should you modify image in place ?
def imcontrast(
        image: NDArray, 
        contrast: float = 1.0,
        gamma: float = 1.0,
        intensity_norm: float = 1.0, 
        blur_size_px: Optional[int] = None, 
        medfilt_size_px: Optional[int] = None
        ) -> NDArray:
    
    if (blur_size_px is not None) and (blur_size_px > 0):
        image = cv2.boxFilter(image, -1, (blur_size_px, blur_size_px))
    image = image/intensity_norm
    if (medfilt_size_px is not None) and (medfilt_size_px > 0):
        image = ndimage.median_filter(image, size = medfilt_size_px)
    image[image<0] = 0
    image = contrast*image**gamma
    image[image>1] = 1
    return image