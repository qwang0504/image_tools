from numpy.typing import NDArray
import numpy as np

def regular_polygon(center: NDArray, n: int, theta: float, scale: int) -> NDArray:
    
    polygon = []
    angle_increment = 2*np.pi/n
    for i in range(n):
        alpha = theta+i*angle_increment
        x, y = np.cos(alpha), np.sin(alpha)
        new_point = center + scale * np.array([x, y])
        polygon.append(new_point)
    
    return np.asarray(polygon, dtype=np.int32)

def star(center: NDArray, n: int, theta: float, scale_0: int, scale_1: int) -> NDArray:
    
    angle_increment = 2*np.pi/n
    polygon_out = regular_polygon(center, n, theta, scale_1)
    polygon_in = regular_polygon(center, n, theta+angle_increment/2, scale_0)
    polygon = [val for pair in zip(polygon_out, polygon_in) for val in pair]

    return np.asarray(polygon, dtype=np.int32)