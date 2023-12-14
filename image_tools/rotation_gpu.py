import numpy as np
from numpy.typing import NDArray
import cv2
from typing import Tuple
from .rotation import Rect, rotation_matrix, translation_matrix, bounding_box_after_rot
# TODO call this transformations and make it more general: crop, translate, resize, rotate, etc

    
# TODO maybe I would like imrotate_GPU to take and return cupy array
# but for now I have issue keeping the memory alive when I create a 
# cupy array from GpuMat inside a function
def imrotate_GPU(image: cv2.cuda.GpuMat, cx: float, cy: float, angle_deg: float) -> Tuple[cv2.cuda.GpuMat, NDArray]:

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
    rotated_image_gpu = cv2.cuda.warpAffine(
        image, 
        warp_mat[:2,:], 
        (bb.width, bb.height), 
        flags=cv2.INTER_NEAREST
    )
    
    # new coordinates of the center of rotation
    new_coords = np.array((cx - bb.left, cy - bb.bottom))

    return rotated_image_gpu, new_coords

