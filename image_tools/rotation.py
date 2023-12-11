import numpy as np
from numpy.typing import NDArray
import cv2
from dataclasses import dataclass
from .convert import cupy_array_to_GpuMat, GpuMat_to_cupy_array
from typing import Tuple

try:
    import cupy as cp
    from cupy.typing import NDArray as CuNDArray
except:
    print('No GPU available, cupy not imported')
    
# TODO call this transformations and make it more general: crop, translate, resize, rotate, etc

@dataclass
class Rect:
    left: int
    bottom: int
    width: int
    height: int

def rotation_matrix(angle_deg: float) -> NDArray:
    angle_rad = np.deg2rad(angle_deg)
    M =  np.array(
        [
            [np.cos(angle_rad), -np.sin(angle_rad), 0],
            [np.sin(angle_rad), np.cos(angle_rad), 0],
            [0,0,1]
        ]
    )
    return M

def translation_matrix(tx,ty):
    return np.array(
        [
            [1, 0, tx],
            [0, 1, ty],
            [0, 0, 1]
        ]
    )

def rotate_vertices(rect: Rect, angle_deg: float) -> NDArray:
    ROI_vertices = np.array(
        [
            [0, 0, 1],
            [0, rect.height, 1],
            [rect.width, rect.height, 1],
            [rect.width, 0, 1],
        ], 
        dtype = np.float32
    )
    rotated_vertices = ROI_vertices
    rotated_vertices = rotated_vertices - np.array([rect.left, rect.bottom, 0])
    rotated_vertices = rotated_vertices @ rotation_matrix(angle_deg) 
    rotated_vertices = rotated_vertices + np.array([rect.left, rect.bottom, 0])
    return rotated_vertices


def bounding_box_after_rot(rect: Rect, angle_deg: float) -> Rect:

    rotated_vertices = rotate_vertices(rect, angle_deg)

    # get bounding box coordinates
    bb_left = np.floor(min(rotated_vertices[:,0]))
    bb_right = np.ceil(max(rotated_vertices[:,0]))
    bb_bottom = np.floor(min(rotated_vertices[:,1]))
    bb_top = np.ceil(max(rotated_vertices[:,1]))
    bb_width = bb_right-bb_left
    bb_height = bb_top-bb_bottom

    return Rect(int(bb_left), int(bb_bottom), int(bb_width), int(bb_height))

def imrotate(image: NDArray, cx: float, cy: float, angle_deg: float) -> Tuple[NDArray, NDArray]:
    # compute bounding box after rotation
    imrect = Rect(cx, cy, image.shape[1], image.shape[0])
    bb = bounding_box_after_rot(imrect, angle_deg)

    # rotate and translate image
    T0 = translation_matrix(-cx, -cy)
    R = rotation_matrix(angle_deg)
    T1 = translation_matrix(cx, cy)
    T2 = translation_matrix(-bb.left, -bb.bottom)
    warp_mat = T2 @ np.linalg.inv(T1 @ R @ T0)
    rotated_image = cv2.warpAffine(
        image, 
        warp_mat[:2,:], 
        (bb.width, bb.height), 
        flags=cv2.INTER_NEAREST
    )
    
    # new coordinates of the center of rotation
    new_coords = np.array((cx - bb.left, cy - bb.bottom))

    return rotated_image, new_coords
    
def imrotate_GPU(image: CuNDArray, cx: float, cy: float, angle_deg: float) -> Tuple[CuNDArray, NDArray]:
#def imrotate_GPU(image: CuNDArray, cx: float, cy: float, angle_deg: float) -> Tuple[cv2.cuda.GpuMat, NDArray]:

    # create GpuMat from cupy ndarray
    image_gpu = cupy_array_to_GpuMat(image)

    # compute bounding box after rotation
    imrect = Rect(cx, cy, image.shape[1], image.shape[0])
    bb = bounding_box_after_rot(imrect, angle_deg)

    # rotate and translate image
    T0 = translation_matrix(-cx, -cy)
    R = rotation_matrix(angle_deg)
    T1 = translation_matrix(cx, cy)
    T2 = translation_matrix(-bb.left, -bb.bottom)
    warp_mat = T2 @ np.linalg.inv(T1 @ R @ T0)
    rotated_image_gpu = cv2.cuda.warpAffine(
        image_gpu, 
        warp_mat[:2,:], 
        (bb.width, bb.height), 
        flags=cv2.INTER_NEAREST
    )
    
    # new coordinates of the center of rotation
    new_coords = np.array((cx - bb.left, cy - bb.bottom))

    return GpuMat_to_cupy_array(rotated_image_gpu), new_coords
    #return rotated_image_gpu, new_coords

# TODO I woudl like to return a cupy array but it looks like the pointers dies outside of 
# the scope of the function