from image_tools import (
    components_size, components_size_GPU,
    bwareaopen, bwareaopen_GPU,
    bwareaclose, bwareaclose_GPU,
    bwareafilter, bwareafilter_GPU,
    bwareaopen_centroids, bwareaopen_centroids_GPU,
    bwareafilter_centroids, bwareafilter_centroids_GPU,
    bwareaopen_props, bwareaopen_props_GPU,
    bwareafilter_props, bwareafilter_props_GPU,
    enhance, enhance_GPU
)
import numpy as np
import cupy as cp
import timeit
import cProfile
import pstats
from pstats import SortKey

# TODO: read that https://docs.cupy.dev/en/stable/user_guide/performance.html

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

## array transfer
t_transfert = timeit.timeit('cu_ar = cp.asarray(ar)', globals=globals(), number=N)*1000/N
print(f'transfert time: {t_transfert:.3f}ms')

## component_size --------------------------------------------------------------------------------------------
with cProfile.Profile() as pr:
    t_cpu_ms = timeit.timeit('(component_sz, ccs) = components_size(ar)', globals=globals(), number=N)*1000/N
    sortby = SortKey.TIME
    ps = pstats.Stats(pr).sort_stats(sortby)
    ps.print_stats(10)

with cProfile.Profile() as pr:
    t_gpu_ms = timeit.timeit('(component_sz, ccs) = components_size_GPU(cu_ar)', globals=globals(), number=N)*1000/N
    sortby = SortKey.TIME
    ps = pstats.Stats(pr).sort_stats(sortby)
    ps.print_stats(10)

print(f'components_size, CPU: {t_cpu_ms:.3f}ms, GPU: {t_gpu_ms:.3f}ms, speedup: {t_cpu_ms/t_gpu_ms:.3f}X')

## bwareaopen --------------------------------------------------------------------------------------------
with cProfile.Profile() as pr:
    t_cpu_ms = timeit.timeit('out = bwareaopen(ar)', globals=globals(), number=N)*1000/N
    sortby = SortKey.TIME
    ps = pstats.Stats(pr).sort_stats(sortby)
    ps.print_stats(10)

with cProfile.Profile() as pr:
    t_gpu_ms = timeit.timeit('out = bwareaopen_GPU(cu_ar)', globals=globals(), number=N)*1000/N
    sortby = SortKey.TIME
    ps = pstats.Stats(pr).sort_stats(sortby)
    ps.print_stats(10)

print(f'bwareaopen, CPU: {t_cpu_ms:.3f}ms, GPU: {t_gpu_ms:.3f}ms, speedup: {t_cpu_ms/t_gpu_ms:.3f}X')

## bwareaclose --------------------------------------------------------------------------------------------
with cProfile.Profile() as pr:
    t_cpu_ms = timeit.timeit('out = bwareaclose(ar)', globals=globals(), number=N)*1000/N
    sortby = SortKey.TIME
    ps = pstats.Stats(pr).sort_stats(sortby)
    ps.print_stats(10)

with cProfile.Profile() as pr:
    t_gpu_ms = timeit.timeit('out = bwareaclose_GPU(cu_ar)', globals=globals(), number=N)*1000/N
    sortby = SortKey.TIME
    ps = pstats.Stats(pr).sort_stats(sortby)
    ps.print_stats(10)

print(f'bwareaclose, CPU: {t_cpu_ms:.3f}ms, GPU: {t_gpu_ms:.3f}ms, speedup: {t_cpu_ms/t_gpu_ms:.3f}X')

## bwareafilter --------------------------------------------------------------------------------------------
with cProfile.Profile() as pr:
    t_cpu_ms = timeit.timeit('out = bwareafilter(ar)', globals=globals(), number=N)*1000/N
    sortby = SortKey.TIME
    ps = pstats.Stats(pr).sort_stats(sortby)
    ps.print_stats(10)

with cProfile.Profile() as pr:
    t_gpu_ms = timeit.timeit('out = bwareafilter_GPU(cu_ar)', globals=globals(), number=N)*1000/N
    sortby = SortKey.TIME
    ps = pstats.Stats(pr).sort_stats(sortby)
    ps.print_stats(10)

print(f'bwareafilter, CPU: {t_cpu_ms:.3f}ms, GPU: {t_gpu_ms:.3f}ms, speedup: {t_cpu_ms/t_gpu_ms:.3f}X')

## bwareaopen_centroids --------------------------------------------------------------------------------------------
with cProfile.Profile() as pr:
    t_cpu_ms = timeit.timeit('out = bwareaopen_centroids(ar)', globals=globals(), number=N)*1000/N
    sortby = SortKey.TIME
    ps = pstats.Stats(pr).sort_stats(sortby)
    ps.print_stats(10)

with cProfile.Profile() as pr:
    t_gpu_ms = timeit.timeit('out = bwareaopen_centroids_GPU(cu_ar)', globals=globals(), number=N)*1000/N
    sortby = SortKey.TIME
    ps = pstats.Stats(pr).sort_stats(sortby)
    ps.print_stats(10)

print(f'bwareaopen_centroids, CPU: {t_cpu_ms:.3f}ms, GPU: {t_gpu_ms:.3f}ms, speedup: {t_cpu_ms/t_gpu_ms:.3f}X')

## bwareafilter_centroids --------------------------------------------------------------------------------------------
with cProfile.Profile() as pr:
    t_cpu_ms = timeit.timeit('out = bwareafilter_centroids(ar)', globals=globals(), number=N)*1000/N
    sortby = SortKey.TIME
    ps = pstats.Stats(pr).sort_stats(sortby)
    ps.print_stats(10)

with cProfile.Profile() as pr:
    t_gpu_ms = timeit.timeit('out = bwareafilter_centroids_GPU(cu_ar)', globals=globals(), number=N)*1000/N
    sortby = SortKey.TIME
    ps = pstats.Stats(pr).sort_stats(sortby)
    ps.print_stats(10)

print(f'bwareafilter_centroids, CPU: {t_cpu_ms:.3f}ms, GPU: {t_gpu_ms:.3f}ms, speedup: {t_cpu_ms/t_gpu_ms:.3f}X')

## bwareaopen_props --------------------------------------------------------------------------------------------
with cProfile.Profile() as pr:
    t_cpu_ms = timeit.timeit('out = bwareaopen_props(ar)', globals=globals(), number=N)*1000/N
    sortby = SortKey.TIME
    ps = pstats.Stats(pr).sort_stats(sortby)
    ps.print_stats(10)

with cProfile.Profile() as pr:
    t_gpu_ms = timeit.timeit('out = bwareaopen_props_GPU(cu_ar)', globals=globals(), number=N)*1000/N
    sortby = SortKey.TIME
    ps = pstats.Stats(pr).sort_stats(sortby)
    ps.print_stats(10)

print(f'bwareaopen_props, CPU: {t_cpu_ms:.3f}ms, GPU: {t_gpu_ms:.3f}ms, speedup: {t_cpu_ms/t_gpu_ms:.3f}X')

## bwareafilter_props --------------------------------------------------------------------------------------------
with cProfile.Profile() as pr:
    t_cpu_ms = timeit.timeit('out = bwareafilter_props(ar)', globals=globals(), number=N)*1000/N
    sortby = SortKey.TIME
    ps = pstats.Stats(pr).sort_stats(sortby)
    ps.print_stats(10)

with cProfile.Profile() as pr:
    t_gpu_ms = timeit.timeit('out = bwareafilter_props_GPU(cu_ar)', globals=globals(), number=N)*1000/N
    sortby = SortKey.TIME
    ps = pstats.Stats(pr).sort_stats(sortby)
    ps.print_stats(10)

print(f'bwareafilter_props, CPU: {t_cpu_ms:.3f}ms, GPU: {t_gpu_ms:.3f}ms, speedup: {t_cpu_ms/t_gpu_ms:.3f}X')

## enhance --------------------------------------------------------------------------------------------------------
N = 100

with cProfile.Profile() as pr:
    t_cpu_ms = timeit.timeit('out = enhance(ar,contrast=1.4,gamma=0.5,brightness=0.2,blur_size_px=10,medfilt_size_px=10)', globals=globals(), number=N)*1000/N
    sortby = SortKey.TIME
    ps = pstats.Stats(pr).sort_stats(sortby)
    ps.print_stats(10)

with cProfile.Profile() as pr:
    t_gpu_ms = timeit.timeit('out = enhance_GPU(cu_ar,contrast=1.4,gamma=0.5,brightness=0.2,blur_size_px=10,medfilt_size_px=10)', globals=globals(), number=N)*1000/N
    sortby = SortKey.TIME
    ps = pstats.Stats(pr).sort_stats(sortby)
    ps.print_stats(10)

print(f'enhance, CPU: {t_cpu_ms:.3f}ms, GPU: {t_gpu_ms:.3f}ms, speedup: {t_cpu_ms/t_gpu_ms:.3f}X')
