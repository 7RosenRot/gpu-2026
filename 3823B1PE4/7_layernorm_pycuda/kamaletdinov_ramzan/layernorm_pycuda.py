import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule

_KERNEL_SRC = r"""
#define THREADS 256
#define WARPS (THREADS / 32)

__device__ __forceinline__ float warp_sum(float v) {
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        v += __shfl_xor_sync(0xffffffffu, v, off);
    return v;
}

__device__ float block_sum(float v, float* smem) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    v = warp_sum(v);
    if (lane == 0) smem[warp] = v;
    __syncthreads();
    if (warp == 0) {
        v = (lane < WARPS) ? smem[lane] : 0.0f;
        v = warp_sum(v);
        if (lane == 0) smem[0] = v;
    }
    __syncthreads();
    const float r = smem[0];
    __syncthreads();
    return r;
}

extern "C" __global__ void layernorm_kernel(const float* __restrict__ x,
                                            const float* __restrict__ gamma,
                                            const float* __restrict__ beta,
                                            float* __restrict__ y,
                                            int row_size, float eps) {
    __shared__ float red[WARPS];
    const float* xr = x + (size_t)blockIdx.x * row_size;
    float* yr = y + (size_t)blockIdx.x * row_size;
    const float inv_n = 1.0f / (float)row_size;

    float s = 0.0f;
    for (int i = threadIdx.x; i < row_size; i += blockDim.x) s += xr[i];
    const float mean = block_sum(s, red) * inv_n;

    float sq = 0.0f;
    for (int i = threadIdx.x; i < row_size; i += blockDim.x) {
        const float d = xr[i] - mean;
        sq += d * d;
    }
    const float var = block_sum(sq, red) * inv_n;
    const float rstd = rsqrtf(var + eps);

    for (int i = threadIdx.x; i < row_size; i += blockDim.x)
        yr[i] = (xr[i] - mean) * rstd * gamma[i] + beta[i];
}
"""

_THREADS = 256
_module = None
_kernel = None

def _get_kernel():
    global _module, _kernel
    if _kernel is None:
        _module = SourceModule(_KERNEL_SRC, no_extern_c=True,
                               options=["-O3", "--use_fast_math"])
        _kernel = _module.get_function("layernorm_kernel")
    return _kernel

def layernorm_pycuda(input, gamma, beta, row_size, eps=1e-5):
    x = np.ascontiguousarray(input, dtype=np.float32).ravel()
    g = np.ascontiguousarray(gamma, dtype=np.float32).ravel()
    b = np.ascontiguousarray(beta, dtype=np.float32).ravel()
    row_size = int(row_size)
    if x.size == 0:
        return np.empty(0, dtype=np.float32)
    if row_size <= 0 or x.size % row_size != 0:
        raise ValueError("input length must be divisible by row_size")
    if g.size != row_size or b.size != row_size:
        raise ValueError("gamma and beta must have length row_size")

    rows = x.size // row_size
    y = np.empty_like(x)

    kernel = _get_kernel()
    kernel(cuda.In(x), cuda.In(g), cuda.In(b), cuda.Out(y),
           np.int32(row_size), np.float32(eps),
           block=(_THREADS, 1, 1), grid=(rows, 1))
    return y
