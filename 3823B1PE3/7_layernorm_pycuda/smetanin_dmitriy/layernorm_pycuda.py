from functools import lru_cache
from operator import index

import numpy as np
import pycuda.driver as cuda
from pycuda import gpuarray
from pycuda.compiler import SourceModule
from pycuda.tools import context_dependent_memoize


_CUDA_SOURCE = r"""
__device__ float BlockSum(float value, float* partial) {
    const unsigned int lane = threadIdx.x;
    partial[lane] = value;
    __syncthreads();

    for (unsigned int offset = blockDim.x / 2; offset > 0; offset /= 2) {
        if (lane < offset) {
            partial[lane] += partial[lane + offset];
        }
        __syncthreads();
    }

    const float sum = partial[0];
    __syncthreads();
    return sum;
}

__global__ void LayerNorm(float* values, const float* gamma, const float* beta,
                          unsigned long long row_size, unsigned long long row_count, float eps) {
    extern __shared__ float partial[];
    const unsigned int lane = threadIdx.x;
    const float inverse_size = 1.0f / static_cast<float>(row_size);

    for (unsigned long long row = blockIdx.x; row < row_count; row += gridDim.x) {
        float* data = values + row * row_size;
        const float first = data[0];
        float sum = 0.0f;
        for (unsigned long long col = lane; col < row_size; col += blockDim.x) {
            sum += data[col] - first;
        }
        const float mean_shifted = BlockSum(sum, partial) * inverse_size;

        float square_sum = 0.0f;
        for (unsigned long long col = lane; col < row_size; col += blockDim.x) {
            const float delta = (data[col] - first) - mean_shifted;
            square_sum = fmaf(delta, delta, square_sum);
        }
        const float variance = BlockSum(square_sum, partial) * inverse_size;
        const float inverse_stddev = rsqrtf(variance + eps);

        for (unsigned long long col = lane; col < row_size; col += blockDim.x) {
            const float delta = (data[col] - first) - mean_shifted;
            data[col] = fmaf(delta * inverse_stddev, gamma[col], beta[col]);
        }
        __syncthreads();
    }
}
"""


@lru_cache(maxsize=1)
def _primary_context():
    return cuda.Device(0).retain_primary_context()


@context_dependent_memoize
def _layernorm_module():
    return SourceModule(_CUDA_SOURCE)


def layernorm_pycuda(input, gamma, beta, row_size, eps=1e-5):
    row_size = index(row_size)
    if row_size <= 0:
        raise ValueError("row_size must be positive")

    values = np.asarray(input, dtype=np.float32)
    scale = np.asarray(gamma, dtype=np.float32)
    offset = np.asarray(beta, dtype=np.float32)
    if values.ndim != 1 or values.size % row_size:
        raise ValueError("input must be a flat matrix with complete rows")
    if scale.shape != (row_size,) or offset.shape != (row_size,):
        raise ValueError("gamma and beta must each contain row_size elements")
    if values.size == 0:
        return np.empty(0, dtype=np.float32)

    cuda.init()
    context = None
    if cuda.Context.get_current() is None:
        context = _primary_context()
        context.push()

    try:
        row_count = values.size // row_size
        threads = min(256, max(32, 1 << (row_size - 1).bit_length()))
        kernel = _layernorm_module().get_function("LayerNorm")
        device_values = gpuarray.to_gpu(np.ascontiguousarray(values))
        device_scale = gpuarray.to_gpu(np.ascontiguousarray(scale))
        device_offset = gpuarray.to_gpu(np.ascontiguousarray(offset))

        kernel(
            device_values,
            device_scale,
            device_offset,
            np.uint64(row_size),
            np.uint64(row_count),
            np.float32(eps),
            block=(threads, 1, 1),
            grid=(min(row_count, 65535), 1, 1),
            shared=threads * values.itemsize,
        )
        return device_values.get()
    finally:
        if context is not None:
            cuda.Context.pop()
