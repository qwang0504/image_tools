import numpy as np
import cv2
from typing import Tuple
from .rotation import Rect, rotation_matrix, translation_matrix, bounding_box_after_rot
import cupy as cp
from cupy.typing import NDArray as CuNDArray
from cucim.skimage import transform

# TODO call this transformations and make it more general: crop, translate, resize, rotate, etc

    
# TODO maybe I would like imrotate_GPU to take and return cupy array
# but for now I have issue keeping the memory alive when I create a 
# cupy array from GpuMat inside a function
def imrotate_GPU(image: cv2.cuda.GpuMat, cx: float, cy: float, angle_deg: float) -> Tuple[cv2.cuda.GpuMat, CuNDArray]:

    w, h = image.size()
    
    # compute bounding box after rotation
    imrect = Rect(cx, cy, w, h)
    bb = bounding_box_after_rot(imrect, angle_deg)

    # rotate and translate image
    T0 = translation_matrix(-cx, -cy)
    R = rotation_matrix(angle_deg)
    T1 = translation_matrix(cx, cy)
    T2 = translation_matrix(-bb.left, -bb.bottom)
    warp_mat = T2 @ np.linalg.inv(T1 @ R @ T0)
    rotated_image_gpu = cv2.cuda.GpuMat(size=(bb.width, bb.height),type=image.type())
    cv2.cuda.warpAffine(
        src=image, 
        dst=rotated_image_gpu,
        M=warp_mat[:2,:], 
        dsize=(bb.width, bb.height), 
        flags=cv2.INTER_NEAREST
    )
    
    # new coordinates of the center of rotation
    new_coords = cp.array((cx - bb.left, cy - bb.bottom))

    return rotated_image_gpu, new_coords

def imrotate_GPU_cucim(image: CuNDArray, cx: float, cy: float, angle_deg: float) -> Tuple[CuNDArray, CuNDArray]:

    w, h = image.shape
    
    # compute bounding box after rotation
    imrect = Rect(cx, cy, w, h)
    bb = bounding_box_after_rot(imrect, angle_deg)

    # rotate and translate image
    T0 = translation_matrix(-cx, -cy)
    R = rotation_matrix(angle_deg)
    T1 = translation_matrix(cx, cy)
    T2 = translation_matrix(-bb.left, -bb.bottom)
    warp_mat = T2 @ np.linalg.inv(T1 @ R @ T0)
    tform = transform.AffineTransform(matrix=cp.asarray(warp_mat))
    rotated_image = transform.warp(image, inverse_map=tform, order=0)
    
    # new coordinates of the center of rotation
    new_coords = cp.array((cx - bb.left, cy - bb.bottom))

    return rotated_image, new_coords