import numpy as np
from scipy import ndimage as ndi
from numpy.typing import NDArray
from typing import Tuple, Optional
from skimage.measure import regionprops

# those are essentially stripped down versions of 
# skimage.morphology.remove_small_objects

def components_size(
        ar: NDArray, 
        connectivity: int = 1
        ) -> Tuple[NDArray, NDArray]:
    
    strel = ndi.generate_binary_structure(ar.ndim, connectivity)
    ccs = np.zeros_like(ar, dtype=np.int32)
    ndi.label(ar, strel, output=ccs)
    component_sz = np.bincount(ccs.ravel()) 
    return (component_sz, ccs)

def bwareaopen(
        ar: NDArray, 
        min_size: int = 64, 
        connectivity: int = 1
        ) -> NDArray:
    
    out = ar.copy()
    component_sz, ccs = components_size(ar, connectivity)
    too_small = component_sz < min_size
    too_small_mask = too_small[ccs]
    out[too_small_mask] = 0

    return out

def bwareaclose(
        ar: NDArray, 
        max_size: int = 256, 
        connectivity: int = 1
        ) -> NDArray:
    
    out = ar.copy()
    component_sz, ccs = components_size(ar, connectivity)
    too_big = component_sz > max_size
    too_big_mask = too_big[ccs]
    out[too_big_mask] = 0

    return out

def bwareafilter(
        ar: NDArray, 
        min_size: int = 64, 
        max_size: int = 256, 
        connectivity: int = 1
        ) -> NDArray:
    
    out = ar.copy()
    component_sz, ccs = components_size(ar, connectivity)
    too_small = component_sz < min_size 
    too_small_mask = too_small[ccs]
    too_big = component_sz > max_size
    too_big_mask = too_big[ccs]
    out[too_small_mask] = 0
    out[too_big_mask] = 0

    return out

def bwareaopen_centroids(
        ar: NDArray, 
        min_size: int = 64,
        connectivity: int = 1,
    ) -> NDArray:

    #label_img = label(ar, connectivity)
    strel = ndi.generate_binary_structure(ar.ndim, connectivity)
    label_img = np.zeros_like(ar, dtype=np.int32)
    ndi.label(ar, strel, output=label_img)
    
    props = regionprops(label_img)
    centroids = []
    for blob in props:
        if blob.area > min_size:
            y, x = blob.centroid
            centroids.append([x, y])
    return np.asarray(centroids, dtype=np.float32)

def bwareafilter_centroids(
        ar: NDArray, 
        min_size: int = 64, 
        max_size: int = 256, 
        min_length: Optional[int] = None,
        max_length: Optional[int] = None,
        min_width: Optional[int] = None,
        max_width: Optional[int] = None,
        connectivity: int = 1
    ) -> NDArray:

    #label_img = label(ar, connectivity)
    strel = ndi.generate_binary_structure(ar.ndim, connectivity)
    label_img = np.zeros_like(ar, dtype=np.int32)
    ndi.label(ar, strel, output=label_img)
    
    props = regionprops(label_img)
    centroids = []
    for blob in props:
        if not (min_size < blob.area < max_size):
            continue
        if (min_width is not None) and (max_width is not None):
            if not (min_width < 2*blob.axis_minor_length < max_width):
                continue
        if (min_length is not None) and (max_length is not None):
            if not (min_length < 2*blob.axis_major_length < max_length):
                continue
        y, x = blob.centroid
        centroids.append([x, y])
    return np.asarray(centroids, dtype=np.float32)

def bwareaopen_props(
        ar: NDArray, 
        min_size: int = 64, 
        connectivity: int = 1
    ):

    #label_img = label(ar, connectivity)
    strel = ndi.generate_binary_structure(ar.ndim, connectivity)
    label_img = np.zeros_like(ar, dtype=np.int32)
    ndi.label(ar, strel, output=label_img)

    props = regionprops(label_img)
    filtered_props = []
    for blob in props:
        if blob.area > min_size:
            filtered_props.append(blob)
    return filtered_props

# OPTIM this is slow, maybe try to run on GPU with CuPy
def bwareafilter_props(
        ar: NDArray, 
        min_size: int = 64, 
        max_size: int = 256, 
        min_length: Optional[int] = None,
        max_length: Optional[int] = None,
        min_width: Optional[int] = None,
        max_width: Optional[int] = None,
        connectivity: int = 1
    ):

    #label_img = label(ar, connectivity)
    strel = ndi.generate_binary_structure(ar.ndim, connectivity)
    label_img = np.zeros_like(ar, dtype=np.int32)
    ndi.label(ar, strel, output=label_img)

    props = regionprops(label_img)
    filtered_props = []
    for blob in props:
        if not (min_size < blob.area < max_size):
            continue
        if (min_width is not None) and (max_width is not None):
            if not (min_width < 2*blob.axis_minor_length < max_width):
                continue
        if (min_length is not None) and (max_length is not None):
            if not (min_length < 2*blob.axis_major_length < max_length):
                continue
        filtered_props.append(blob)
    return filtered_props