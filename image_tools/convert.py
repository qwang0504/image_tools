import numpy as np
from numpy.typing import NDArray
import cv2

try:
    import cupy as cp
    from cupy.typing import NDArray as CuNDArray
except:
    print('No GPU available, cupy not imported')

CUPY_TO_CVTYPE = {
    cp.dtype('uint8'): cv2.CV_8U,
    cp.dtype('int8'): cv2.CV_8S,
    cp.dtype('uint16'): cv2.CV_16U,
    cp.dtype('int16'): cv2.CV_16S,
    cp.dtype('float16'): cv2.CV_16F,
    cp.dtype('int32'): cv2.CV_32S, 
    cp.dtype('float32'): cv2.CV_32F,
    cp.dtype('float64'): cv2.CV_64F,
}

CVTYPE_TO_CUPY = {
    cv2.CV_8U: cp.dtype('uint8'),
    cv2.CV_8S: cp.dtype('int8'),
    cv2.CV_16U: cp.dtype('uint16'),
    cv2.CV_16S: cp.dtype('int16'),
    cv2.CV_16F: cp.dtype('float16'),
    cv2.CV_32S: cp.dtype('int32'), 
    cv2.CV_32F: cp.dtype('float32'),
    cv2.CV_64F: cp.dtype('float64'),
}

def cupy_array_to_GpuMat(image: CuNDArray) -> cv2.cuda.GpuMat:
    
    if len(image.shape) > 3:
        raise ValueError('cupy_array_to_GpuMat::Image has too many dimensions, max 3')
    else if len(image.shape) == 3:
        num_channels = image.shape[2]
    else if len(image.shape) == 2:
        num_channels = 1
    else:
        raise ValueError('cupy_array_to_GpuMat::Image has too few dimensions, min 2')

    return cv2.cuda.createGpuMatFromCudaMemory(
        image.shape, 
        cv2.CV_MAKETYPE(CUPY_TO_CVTYPE[image.dtype], num_channels), 
        image.data.ptr
    )

def GpuMat_to_cupy_array(image: cv2.cuda.GpuMat) -> CuNDArray:

    sz = image.size()
    channels = image.channels()
    num_bytes = sz[0]*sz[1]*channels*image.elemSize()
    mem = cupy.cuda.UnownedMemory(image.cudaPtr(), num_bytes, owner=None)
    memptr = cupy.cuda.MemoryPointer(mem, offset=0)
    if channels > 1:
        arr_sz = (sz[0], sz[1], image.channels())
    else:
        arr_sz = sz
    return cupy.ndarray(arr_sz, dtype=CVTYPE_TO_CUPY[image.type()], memptr=memptr)    

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
        single_image = input_image.astype(np.float32)

    elif input_image.dtype == np.bool_:
        single_image = input_image.astype(np.float32) 

    else:
        raise ValueError('wrong image type, cannot convert to single')

    return single_image

def im2single_GPU(input_image: CuNDArray) -> CuNDArray:
    """
    Transform input image into a single precision floating point image
    """

    if cp.issubdtype(input_image.dtype, cp.integer):
        # if integer type, transform to float and scale between 0 and 1
        ui_info = cp.iinfo(input_image.dtype)
        single_image = input_image.astype(cp.float32) / ui_info.max

    elif cp.issubdtype(input_image.dtype, cp.floating):
        # if already a floating type, convert to single precision
        single_image = input_image.astype(cp.float32)

    elif input_image.dtype == cp.bool_:
        single_image = input_image.astype(cp.float32) 

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
        double_image = input_image.astype(np.float64)

    elif input_image.dtype == np.bool_:
        double_image = input_image.astype(np.float64) 

    else:
        raise ValueError('wrong image type, cannot convert to double')
    
    return double_image

def im2uint8(input_image: NDArray) -> NDArray:

    if np.issubdtype(input_image.dtype, np.integer):
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