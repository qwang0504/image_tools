import numpy as np
from scipy import ndimage as ndi
from numpy.typing import NDArray
from typing import Tuple, Optional, List
from skimage.measure import regionprops
from skimage.measure._regionprops import _RegionProperties
import cv2

# those are essentially stripped down versions of 
# skimage.morphology.remove_small_objects

def label(        
        ar: NDArray, 
        connectivity: int = 1
    ) -> NDArray:

    strel = ndi.generate_binary_structure(ar.ndim, connectivity)
    return ndi.label(ar, strel)[0]

def properties(
        ar: NDArray, 
        connectivity: int = 1
    ) -> List[_RegionProperties]:

    label_img = label(ar, connectivity)
    return regionprops(label_img)

def components_size(
        ar: NDArray, 
        connectivity: int = 1
        ) -> Tuple[NDArray, NDArray]:
    
    ccs = label(ar, connectivity)
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
    
    props = properties(ar, connectivity)
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
        if (min_width is not None) and (max_width is not None) and (max_width > 0):
            if not (min_width < 2*blob.axis_minor_length < max_width):
                continue
        if (min_length is not None) and (max_length is not None)  and (max_length > 0):
            if not (min_length < 2*blob.axis_major_length < max_length):
                continue
        y, x = blob.centroid
        centroids.append([x, y])
    return np.asarray(centroids, dtype=np.float32)

def bwareafilter_centroid_cv2(
        ar: NDArray, 
        min_size: int = 64, 
        max_size: int = 256, 
        min_length: Optional[int] = None,
        max_length: Optional[int] = None,
        min_width: Optional[int] = None,
        max_width: Optional[int] = None,
        connectivity: int = 4
    ):
    # note that width and length are not treated equivalently to bwareafilter_centroids
    # here it is the width and height of the bounding box instead of minor and major axis

    n_components, labels, stats, centroids = cv2.connectedComponentsWithStats(
        ar,
        connectivity = connectivity,
        cv2.CV_32S
    )
    kept_centroids = []
    for c in range(n_components):
        if not (min_size < stats[c, cv2.CC_STAT_AREA] < max_size):
            continue
        if (min_width is not None) and (max_width is not None) and (max_width > 0):
            if not (min_width < stats[c, cv2.CC_STAT_WIDTH] < max_width):
                continue
        if (min_length is not None) and (max_length is not None)  and (max_length > 0):
            if not (min_length < stats[i, cv2.CC_STAT_HEIGHT]< max_length):
                continue
        kept_centroids.append(centroids[c])
        
    return np.asarray(centroids, dtype=np.float32)

def bwareaopen_props(
        ar: NDArray, 
        min_size: int = 64, 
        connectivity: int = 1
    ) -> list:

    props = properties(ar, connectivity)
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
        if (min_width is not None) and (max_width is not None) and (max_width > 0):
            if not (min_width < 2*blob.axis_minor_length < max_width):
                continue
        if (min_length is not None) and (max_length is not None) and (max_length > 0):
            if not (min_length < 2*blob.axis_major_length < max_length):
                continue
        filtered_props.append(blob)
    return filtered_props

