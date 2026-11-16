import operator
import threading

import numpy as np
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
from pycuda.tools import context_dependent_memoize


_source = r"""
#include <cuda_runtime.h>
#include <stddef.h>

__device__ float block_sum(float value, float* scratch) {
    const int tid = threadIdx.x;
    scratch[tid] = value;
    __syncthreads();
    for (int offset = 128; offset > 0; offset /= 2) {
        if (tid < offset) {
            scratch[tid] += scratch[tid + offset];
        }
        __syncthreads();
    }
    const float result = scratch[0];
    __syncthreads();
    return result;
}

extern "C" __global__ void layer_norm(
    const float* input, const float* gamma, const float* beta,
    float* output, int row_size, float eps) {
    __shared__ float scratch[256];
    const int tid = threadIdx.x;
    const size_t start = static_cast<size_t>(blockIdx.x) * row_size;
    const float shift = input[start];
    float values[ITEMS > 0 ? ITEMS : 1];
    float sum = 0.0f;

#if ITEMS > 0
#pragma unroll
    for (int i = 0; i < ITEMS; ++i) {
        const int col = tid + i * 256;
        values[i] = col < row_size ? input[start + col] - shift : 0.0f;
        sum += values[i];
    }
#else
    for (size_t col = tid; col < static_cast<size_t>(row_size); col += 256) {
        sum += input[start + col] - shift;
    }
#endif
    const float mean = block_sum(sum, scratch) / row_size;
    float squares = 0.0f;

#if ITEMS > 0
#pragma unroll
    for (int i = 0; i < ITEMS; ++i) {
        const int col = tid + i * 256;
        if (col < row_size) {
            values[i] -= mean;
            squares += values[i] * values[i];
        }
    }
#else
    for (size_t col = tid; col < static_cast<size_t>(row_size); col += 256) {
        const float difference = (input[start + col] - shift) - mean;
        squares += difference * difference;
    }
#endif
    const float variance = block_sum(squares, scratch) / row_size;
    const float scale = rsqrtf(variance + eps);

#if ITEMS > 0
#pragma unroll
    for (int i = 0; i < ITEMS; ++i) {
        const int col = tid + i * 256;
        if (col < row_size) {
            output[start + col] = values[i] * scale * gamma[col] + beta[col];
        }
    }
#else
    for (size_t col = tid; col < static_cast<size_t>(row_size); col += 256) {
        const float difference = (input[start + col] - shift) - mean;
        output[start + col] = difference * scale * gamma[col] + beta[col];
    }
#endif
}
"""


class _Workspace:
    def __init__(self):
        self.buffers = [None] * 4
        self.capacities = [0] * 4
        self.kernels = {}
        self.lock = threading.Lock()

    def reserve(self, sizes):
        for i, size in enumerate(sizes):
            if size > self.capacities[i]:
                buffer = cuda.mem_alloc(size)
                self.buffers[i] = buffer
                self.capacities[i] = size


_initialization_lock = threading.Lock()


@context_dependent_memoize
def _get_workspace():
    return _Workspace()


def _runtime():
    with _initialization_lock:
        cuda.init()
        if cuda.Context.get_current() is None:
            import pycuda.autoinit

        if cuda.Context.get_current() is None:
            raise RuntimeError("A current CUDA context is required")
        return _get_workspace()


def layernorm_pycuda(input, gamma, beta, row_size, eps=1e-5):
    row_size = operator.index(row_size)
    if row_size <= 0 or row_size > np.iinfo(np.int32).max:
        raise ValueError("row_size must be a positive int32 value")
    eps = float(eps)
    if not np.isfinite(eps) or eps <= 0 or eps > np.finfo(np.float32).max:
        raise ValueError("eps must be positive and finite in float32")
    eps = np.float32(eps)
    if eps == 0:
        raise ValueError("eps is too small for float32")

    x = np.ascontiguousarray(input, dtype=np.float32).reshape(-1)
    gamma = np.ascontiguousarray(gamma, dtype=np.float32).reshape(-1)
    beta = np.ascontiguousarray(beta, dtype=np.float32).reshape(-1)
    if x.size % row_size != 0:
        raise ValueError("Input size must be divisible by row_size")
    if gamma.size != row_size or beta.size != row_size:
        raise ValueError("gamma and beta must contain row_size elements")
    if x.size == 0:
        return np.empty(0, dtype=np.float32)
    rows = x.size // row_size
    if rows > np.iinfo(np.int32).max:
        raise ValueError("Too many rows for the CUDA grid")

    items = next((n for n in (1, 4, 16, 64) if row_size <= 256 * n), 0)
    workspace = _runtime()

    with workspace.lock:
        if items not in workspace.kernels:
            module = SourceModule(
                _source, options=["-O3", "-DITEMS=" + str(items)],
                no_extern_c=True,
            )
            workspace.kernels[items] = (module, module.get_function("layer_norm"))
        kernel = workspace.kernels[items][1]
        workspace.reserve((x.nbytes, gamma.nbytes, beta.nbytes, x.nbytes))
        d_input, d_gamma, d_beta, d_output = workspace.buffers
        cuda.memcpy_htod(d_input, x)
        cuda.memcpy_htod(d_gamma, gamma)
        cuda.memcpy_htod(d_beta, beta)
        kernel(
            d_input, d_gamma, d_beta, d_output, np.int32(row_size), eps,
            block=(256, 1, 1), grid=(rows, 1),
        )
        output = np.empty_like(x)
        cuda.memcpy_dtoh(output, d_output)
    return output
