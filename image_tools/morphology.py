import numpy as np
from scipy import ndimage as ndi
from numpy.typing import NDArray
from typing import Tuple, Optional
from skimage.measure import regionprops

try:
    import cupy as cp
    from cupy.typing import NDArray as CuNDArray
    from cupyx.scipy import ndimage as cu_ndi
    from cucim.skimage.measure import regionprops as cu_regionprops
except:
    print('No GPU available, cupy not imported')

# those are essentially stripped down versions of 
# skimage.morphology.remove_small_objects

def label(        
        ar: NDArray, 
        connectivity: int = 1
    ) -> NDArray:

    strel = ndi.generate_binary_structure(ar.ndim, connectivity)
    return ndi.label(ar, strel)[0]

def label_GPU(        
        ar: CuNDArray, 
        connectivity: int = 1
    ) -> CuNDArray:

    strel = cu_ndi.generate_binary_structure(ar.ndim, connectivity)
    return cu_ndi.label(ar, strel)[0]

def properties(
        ar: NDArray, 
        connectivity: int = 1
    ) -> list:

    label_img = label(ar, connectivity)
    return regionprops(label_img)

def properties_GPU(
        ar: CuNDArray, 
        connectivity: int = 1
    ) -> list:

    label_img = label_GPU(ar, connectivity)
    return cu_regionprops(label_img)

def components_size(
        ar: NDArray, 
        connectivity: int = 1
        ) -> Tuple[NDArray, NDArray]:
    
    ccs = label(ar, connectivity)
    component_sz = np.bincount(ccs.ravel()) 
    return (component_sz, ccs)

def components_size_GPU(
        ar: CuNDArray, 
        connectivity: int = 1
        ) -> Tuple[CuNDArray, CuNDArray]:
    
    ccs = label_GPU(ar, connectivity)
    component_sz = cp.bincount(ccs.ravel()) 
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

def bwareaopen_GPU(
        ar: CuNDArray, 
        min_size: int = 64, 
        connectivity: int = 1
        ) -> CuNDArray:
    
    out = ar.copy()
    component_sz, ccs = components_size_GPU(ar, connectivity)
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

def bwareaclose_GPU(
        ar: CuNDArray, 
        max_size: int = 256, 
        connectivity: int = 1
        ) -> CuNDArray:
    
    out = ar.copy()
    component_sz, ccs = components_size_GPU(ar, connectivity)
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

def bwareafilter_GPU(
        ar: CuNDArray, 
        min_size: int = 64, 
        max_size: int = 256, 
        connectivity: int = 1
        ) -> CuNDArray:
    
    out = ar.copy()
    component_sz, ccs = components_size_GPU(ar, connectivity)
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
    
    props = properties(ar, connectivity)
    centroids = [blob.centroid[::-1] for blob in props if blob.area > min_size]
    return np.asarray(centroids, dtype=np.float32)

def bwareaopen_centroids_GPU(
        ar: CuNDArray, 
        min_size: int = 64,
        connectivity: int = 1,
    ) -> CuNDArray:

    props = properties_GPU(ar, connectivity)
    centroids = [blob.centroid[::-1] for blob in props if blob.area > min_size]
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

    props = properties(ar, connectivity)
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

def bwareafilter_centroids_GPU(
        ar: CuNDArray, 
        min_size: int = 64, 
        max_size: int = 256, 
        min_length: Optional[int] = None,
        max_length: Optional[int] = None,
        min_width: Optional[int] = None,
        max_width: Optional[int] = None,
        connectivity: int = 1
    ) -> CuNDArray:

    props = properties_GPU(ar, connectivity)
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
    return cp.asarray(centroids, dtype=np.float32)

def bwareaopen_props(
        ar: NDArray, 
        min_size: int = 64, 
        connectivity: int = 1
    ) -> list:

    props = properties(ar, connectivity)
    return [blob for blob in props if blob.area > min_size]

def bwareaopen_props_GPU(
        ar: CuNDArray, 
        min_size: int = 64, 
        connectivity: int = 1
    ) -> list:

    props = properties_GPU(ar, connectivity)
    return [blob for blob in props if blob.area > min_size]

def bwareafilter_props(
        ar: NDArray, 
        min_size: int = 64, 
        max_size: int = 256, 
        min_length: Optional[int] = None,
        max_length: Optional[int] = None,
        min_width: Optional[int] = None,
        max_width: Optional[int] = None,
        connectivity: int = 1
    ) -> list:

    props = properties(ar, connectivity)
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

def bwareafilter_props_GPU(
        ar: CuNDArray, 
        min_size: int = 64, 
        max_size: int = 256, 
        min_length: Optional[int] = None,
        max_length: Optional[int] = None,
        min_width: Optional[int] = None,
        max_width: Optional[int] = None,
        connectivity: int = 1
    ) -> list:

    props = properties_GPU(ar, connectivity)
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
