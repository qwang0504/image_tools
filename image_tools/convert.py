import numpy as np
from numpy.typing import NDArray

def im2single(input_image: NDArray) -> NDArray:
    """
    Transform input image into a single precision floating point image
    """

    if np.issubdtype(input_image.dtype, np.integer):
        # if integer type, transform to float and scale between 0 and 1
        ui_info = np.iinfo(input_image.dtype)
        single_image = input_image.astype(np.float32) / ui_info.max

    elif np.issubdtype(input_image.dtype, np.floating):
        # if already a floating type, convert to single precision
        if input_image.dtype == np.float32:
            return input_image
        
        single_image = input_image.astype(np.float32)

    elif input_image.dtype == np.bool_:
        single_image = input_image.astype(np.float32) 

    else:
        raise ValueError('wrong image type, cannot convert to single')

    return single_image

def im2double(input_image: NDArray) -> NDArray:
    """
    Transform input image into a double precision floating point image
    """

    if np.issubdtype(input_image.dtype, np.integer):
        # if integer type, transform to float and scale between 0 and 1
        ui_info = np.iinfo(input_image.dtype)
        double_image = input_image.astype(np.float64) / ui_info.max

    elif np.issubdtype(input_image.dtype, np.floating):
        # if already a floating type, convert to double precision
        if input_image.dtype == np.float64:
            return input_image
        
        double_image = input_image.astype(np.float64)

    elif input_image.dtype == np.bool_:
        double_image = input_image.astype(np.float64) 

    else:
        raise ValueError('wrong image type, cannot convert to double')
    
    return double_image

def im2uint8(input_image: NDArray) -> NDArray:
    '''Convert image to uint8. Note that this is slow for large images'''
    
    if np.issubdtype(input_image.dtype, np.integer):
        if input_image.dtype == np.uint8:
            return input_image
        
        ui_info = np.iinfo(input_image.dtype)
        uint8_image = (input_image *  255.0/ui_info.max).astype(np.uint8)

    elif np.issubdtype(input_image.dtype, np.floating):
        uint8_image = (input_image *  255.0).astype(np.uint8)

    elif input_image.dtype == np.bool_:
        uint8_image = 255 * input_image.astype(np.uint8) 

    else:
        raise ValueError('wrong image type, cannot convert to uint8')

    return uint8_image
   
def im2gray(input_image: NDArray) -> NDArray:
    """
    Transform color input into grayscale by taking only the first channel

    Inputs:
        input_image: M x N x C | M x N x C x K numpy array 

    Outputs:
        M x N | M x N x K numpy array 
    """

    shp = input_image.shape

    if len(shp) == 2:
        # already grayscale, nothing to do
        return input_image
    
    if len(shp) == 3:
        # M x N X C
        return input_image[:,:,0]
    
    else:
        raise ValueError('wrong image type, cannot convert to grayscale')
    
def im2rgb(input_image: NDArray) -> NDArray:
    """
    Transform grayscale input into color image
    """

    shp = input_image.shape

    if len(shp) == 3 and shp[2] == 3:
        # already RGB, nothing to do
        return input_image
    
    elif len(shp) == 2:
        rgb_image = np.dstack((input_image, input_image, input_image))
        return rgb_image
    
    else:
        raise ValueError('wrong image type, cannot convert to RGB')
    
