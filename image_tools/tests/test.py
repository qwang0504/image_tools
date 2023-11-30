from image_tools import (
    components_size, components_size_GPU,
    bwareaopen, bwareaopen_GPU,
    bwareaclose, bwareaclose_GPU,
    bwareafilter, bwareafilter_GPU,
    bwareaopen_centroids, bwareaopen_centroids_GPU,
    bwareafilter_centroids, bwareafilter_centroids_GPU,
    bwareaopen_props, bwareaopen_props_GPU,
    bwareafilter_props, bwareafilter_props_GPU
)
import numpy as np
import cupy as cp
import timeit

# dummy array creation
SZ = 2048

# CPU array
ar = np.zeros((SZ,SZ), dtype=np.float32)
X,Y = np.mgrid[0:SZ,0:SZ]
X0 = np.random.randint(0,2048,size=(100,), dtype=int)
Y0 = np.random.randint(0,2048,size=(100,), dtype=int)
R = np.random.randint(10,100,size=(100,), dtype=int) 
for x0, y0, r in zip(X0, Y0, R):
    circle = (X - x0)**2 + (Y - y0)**2 <= r**2
    ar[circle] = 1

# GPU array
cu_ar = cp.asarray(ar)

# number of repetitions
N = 1000

## component_size --------------------------------------------------------------------------------------------
t_cpu_ms = timeit.timeit('(component_sz, ccs) = components_size(ar)', globals=globals(), number=N)*1000/N
t_gpu_ms = timeit.timeit('(component_sz, ccs) = components_size_GPU(cu_ar)', globals=globals(), number=N)*1000/N
print(f'components_size, CPU: {t_cpu_ms:.3f}ms, GPU: {t_gpu_ms:.3f}ms, speedup: {t_cpu_ms/t_gpu_ms:.3f}X')

## bwareaopen --------------------------------------------------------------------------------------------
t_cpu_ms = timeit.timeit('out = bwareaopen(ar)', globals=globals(), number=N)*1000/N
t_gpu_ms = timeit.timeit('out = bwareaopen_GPU(cu_ar)', globals=globals(), number=N)*1000/N
print(f'bwareaopen, CPU: {t_cpu_ms:.3f}ms, GPU: {t_gpu_ms:.3f}ms, speedup: {t_cpu_ms/t_gpu_ms:.3f}X')

## bwareaclose --------------------------------------------------------------------------------------------
t_cpu_ms = timeit.timeit('out = bwareaclose(ar)', globals=globals(), number=N)*1000/N
t_gpu_ms = timeit.timeit('out = bwareaclose_GPU(cu_ar)', globals=globals(), number=N)*1000/N
print(f'bwareaopen, CPU: {t_cpu_ms:.3f}ms, GPU: {t_gpu_ms:.3f}ms, speedup: {t_cpu_ms/t_gpu_ms:.3f}X')

## bwareafilter --------------------------------------------------------------------------------------------
t_cpu_ms = timeit.timeit('out = bwareafilter(ar)', globals=globals(), number=N)*1000/N
t_gpu_ms = timeit.timeit('out = bwareafilter_GPU(cu_ar)', globals=globals(), number=N)*1000/N
print(f'bwareaopen, CPU: {t_cpu_ms:.3f}ms, GPU: {t_gpu_ms:.3f}ms, speedup: {t_cpu_ms/t_gpu_ms:.3f}X')

## bwareaopen_centroids --------------------------------------------------------------------------------------------
t_cpu_ms = timeit.timeit('out = bwareaopen_centroids(ar)', globals=globals(), number=N)*1000/N
t_gpu_ms = timeit.timeit('out = bwareafilter_GPU(cu_ar)', globals=globals(), number=N)*1000/N
print(f'bwareaopen_centroids, CPU: {t_cpu_ms:.3f}ms, GPU: {t_gpu_ms:.3f}ms, speedup: {t_cpu_ms/t_gpu_ms:.3f}X')

## bwareafilter_centroids --------------------------------------------------------------------------------------------
t_cpu_ms = timeit.timeit('out = bwareafilter_centroids(ar)', globals=globals(), number=N)*1000/N
t_gpu_ms = timeit.timeit('out = bwareafilter_centroids_GPU(cu_ar)', globals=globals(), number=N)*1000/N
print(f'bwareafilter_centroids, CPU: {t_cpu_ms:.3f}ms, GPU: {t_gpu_ms:.3f}ms, speedup: {t_cpu_ms/t_gpu_ms:.3f}X')

## bwareaopen_props --------------------------------------------------------------------------------------------
t_cpu_ms = timeit.timeit('out = bwareaopen_props(ar)', globals=globals(), number=N)*1000/N
t_gpu_ms = timeit.timeit('out = bwareaopen_props_GPU(cu_ar)', globals=globals(), number=N)*1000/N
print(f'bwareaopen_props, CPU: {t_cpu_ms:.3f}ms, GPU: {t_gpu_ms:.3f}ms, speedup: {t_cpu_ms/t_gpu_ms:.3f}X')

## bwareafilter_props --------------------------------------------------------------------------------------------
t_cpu_ms = timeit.timeit('out = bwareafilter_props(ar)', globals=globals(), number=N)*1000/N
t_gpu_ms = timeit.timeit('out = bwareafilter_props_GPU(cu_ar)', globals=globals(), number=N)*1000/N
print(f'bwareafilter_props, CPU: {t_cpu_ms:.3f}ms, GPU: {t_gpu_ms:.3f}ms, speedup: {t_cpu_ms/t_gpu_ms:.3f}X')
