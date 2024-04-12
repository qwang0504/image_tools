
import cv2
import cupy as cp
from cupy.typing import NDArray as CuNDArray

CUPY_TO_CVTYPE = {
    cp.dtype('uint8'): cv2.CV_8U,
    cp.dtype('int8'): cv2.CV_8S,
    cp.dtype('uint16'): cv2.CV_16U,
    cp.dtype('int16'): cv2.CV_16S,
    cp.dtype('int32'): cv2.CV_32S, 
    cp.dtype('float16'): cv2.CV_16F,
    cp.dtype('float32'): cv2.CV_32F,
    cp.dtype('float64'): cv2.CV_64F,
}

CVTYPE_TO_CUPY = {
    cv2.CV_8U: cp.dtype('uint8'),
    cv2.CV_8S: cp.dtype('int8'),
    cv2.CV_16U: cp.dtype('uint16'),
    cv2.CV_16S: cp.dtype('int16'),
    cv2.CV_32S: cp.dtype('int32'), 
    cv2.CV_16F: cp.dtype('float16'),
    cv2.CV_32F: cp.dtype('float32'),
    cv2.CV_64F: cp.dtype('float64'),
}

def cupy_array_to_GpuMat(image: CuNDArray) -> cv2.cuda.GpuMat:
    
    if len(image.shape) > 3:
        raise ValueError('cupy_array_to_GpuMat::Image has too many dimensions, max 3')
    elif len(image.shape) == 3:
        h, w, num_channels = image.shape
    elif len(image.shape) == 2:
        h, w = image.shape
        num_channels = 1
    else:
        raise ValueError('cupy_array_to_GpuMat::Image has too few dimensions, min 2')

    gpu_mat = cv2.cuda.createGpuMatFromCudaMemory(
        (w*num_channels, h), 
        cv2.CV_MAKETYPE(CUPY_TO_CVTYPE[image.dtype], num_channels), 
        image.data.ptr
    )
    # this creates a continuous matrix, which I don't want (problem with warpaffine).
    # copying on device seems to get rid of that. That should have a moderate impact on performance
    gpu_noncontinuous = gpu_mat.copyTo()
  
    return gpu_noncontinuous

def GpuMat_to_cupy_array(image: cv2.cuda.GpuMat) -> CuNDArray:

    w, h = image.size()
    channels = image.channels()
    gap = image.step1() - w
    num_bytes = (w+gap)*h*channels*image.elemSize()
    mem = cp.cuda.UnownedMemory(image.cudaPtr(), num_bytes, owner=None)
    memptr = cp.cuda.MemoryPointer(mem, offset=0)
    if channels > 1:
        arr_sz = (h, w+gap, image.channels())
    else:
        arr_sz = (h, w+gap)
    arr = cp.ndarray(arr_sz, dtype=CVTYPE_TO_CUPY[image.type()], memptr=memptr)  
    
    return arr[:h,:w] 

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

def im2gray_GPU(input_image: CuNDArray) -> CuNDArray:
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
    
    if len(shp) >= 3:
        return input_image[...,0]
    
    else:
        raise ValueError('wrong image type, cannot convert to grayscale')
    

def im2uint8_GPU(input_image: CuNDArray) -> CuNDArray:
    '''Convert image to uint8.'''
    
    if cp.issubdtype(input_image.dtype, cp.integer):
        if input_image.dtype == cp.uint8:
            return input_image
        
        ui_info = cp.iinfo(input_image.dtype)
        uint8_image = (input_image *  255.0/ui_info.max).astype(cp.uint8)

    elif cp.issubdtype(input_image.dtype, cp.floating):
        uint8_image = (input_image *  255.0).astype(cp.uint8)

    elif input_image.dtype == cp.bool_:
        uint8_image = 255 * input_image.astype(cp.uint8) 

    else:
        raise ValueError('wrong image type, cannot convert to uint8')

    return uint8_image

def im2double_GPU(input_image: CuNDArray) -> CuNDArray:
    """
    Transform input image into a double precision floating point image
    """

    if cp.issubdtype(input_image.dtype, cp.integer):
        # if integer type, transform to float and scale between 0 and 1
        ui_info = cp.iinfo(input_image.dtype)
        double_image = input_image.astype(cp.float64) / ui_info.max

    elif cp.issubdtype(input_image.dtype, cp.floating):
        # if already a floating type, convert to double precision
        if input_image.dtype == cp.float64:
            return input_image
        
        double_image = input_image.astype(cp.float64)

    elif input_image.dtype == cp.bool_:
        double_image = input_image.astype(cp.float64) 

    else:
        raise ValueError('wrong image type, cannot convert to double')
    
    return double_image