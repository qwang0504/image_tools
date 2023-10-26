import numpy as np
from numpy.typing import NDArray
import cv2
from geometry.rect import Rect

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
        
def diagonal_crop(image: NDArray, rect: Rect, angle_deg: float) -> NDArray:
    # compute bounding box after rotation
    imrect = Rect(rect.left, rect.bottom, image.shape[1], image.shape[0])
    bb = bounding_box_after_rot(imrect, angle_deg)

    # rotate and translate image
    T0 = translation_matrix(-rect.left, -rect.bottom)
    R = rotation_matrix(angle_deg)
    T1 = translation_matrix(rect.left, rect.bottom)
    T2 = translation_matrix(-bb.left, -bb.bottom)
    warp_mat = T2 @ np.linalg.inv(T1 @ R @ T0)
    rotated_image = cv2.warpAffine(image, warp_mat[:2,:], (bb.width, bb.height), flags=cv2.INTER_NEAREST)
    
    # crop rotated image        
    left = rect.left - bb.left
    bottom = rect.bottom - bb.bottom
    right = left + rect.width
    top = bottom + rect.height

    crop_bottom = max(0, bottom)
    crop_left = max(0, left)
    crop_top = min(top, rotated_image.shape[0])
    crop_right = min(right, rotated_image.shape[1])
    rotated_crop = rotated_image[crop_bottom:crop_top, crop_left:crop_right]

    # pad image if necessary 
    if left < 0: 
        pad_h = (-left, left + rect.width - rotated_crop.shape[1])
    else:
        pad_h = (0, rect.width - rotated_crop.shape[1])
    if bottom < 0:
        pad_v = (-bottom, bottom + rect.height - rotated_crop.shape[0])
    else:
        pad_v = (0, rect.height - rotated_crop.shape[0])
    rotated_crop = np.pad(rotated_crop,(pad_v,pad_h), constant_values=0)

    return rotated_crop

def imrotate(image: NDArray, cx: float, cy: float, angle_deg: float) -> NDArray:
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
    