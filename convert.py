import numpy as np
from numpy.typing import NDArray

def im2single(input_image: NDArray) -> NDArray:
    """
    Transform input image into a single precision floating point image
    """
    ui_info = np.iinfo(input_image.dtype)
    return input_image.astype(np.float32) / ui_info.max

def im2gray(input_image: NDArray) -> NDArray:
    """
    Transform color input into grayscale by taking only the first channel

    Inputs:
        input_image: M x N x C | M x N x C x K numpy array 

    Outputs:
        M x N | M x N x K numpy array 
    """

    if len(input_image.shape) == 3:
        # M x N X C
        return input_image[:,:,0]
    elif len(input_image.shape) == 4:
        # M x N X C x K
        return np.squeeze(input_image[:,:,0,:])