import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule

_THREADS = 256

_KERNEL_SOURCE = r"""
#define THREADS 256

__device__ float block_sum(float value, float* shared)
{
    shared[threadIdx.x] = value;
    __syncthreads();
    for (int stride = THREADS / 2; stride > 0; stride /= 2) {
        if (threadIdx.x < stride) {
            shared[threadIdx.x] += shared[threadIdx.x + stride];
        }
        __syncthreads();
    }
    float result = shared[0];
    __syncthreads();
    return result;
}

__global__ void layernorm_kernel(const float* input,
                                 const float* gamma,
                                 const float* beta,
                                 float* output,
                                 int row_size,
                                 float eps)
{
    __shared__ float shared[THREADS];

    const float* in_row = input + (size_t)blockIdx.x * row_size;
    float* out_row = output + (size_t)blockIdx.x * row_size;

    float local = 0.0f;
    for (int i = threadIdx.x; i < row_size; i += THREADS) {
        local += in_row[i];
    }
    const float mean = block_sum(local, shared) / row_size;

    local = 0.0f;
    for (int i = threadIdx.x; i < row_size; i += THREADS) {
        const float d = in_row[i] - mean;
        local += d * d;
    }
    const float variance = block_sum(local, shared) / row_size;
    const float inv_std = rsqrtf(variance + eps);

    for (int i = threadIdx.x; i < row_size; i += THREADS) {
        out_row[i] = (in_row[i] - mean) * inv_std * gamma[i] + beta[i];
    }
}
"""

_kernel = None


def _get_kernel():
    global _kernel
    if _kernel is None:
        module = SourceModule(_KERNEL_SOURCE)
        _kernel = module.get_function("layernorm_kernel")
    return _kernel


def layernorm_pycuda(input, gamma, beta, row_size, eps=1e-5):
    """
    Apply Layer Normalization to each row of the input matrix.

    Parameters
    ----------
    input : list or numpy.ndarray of float
        Flattened matrix in row-major order. Its length must be divisible by row_size.
    gamma : list or numpy.ndarray of float
        Scale parameter, length = row_size.
    beta : list or numpy.ndarray of float
        Shift parameter, length = row_size.
    row_size : int
        Number of features per row (i.e., number of columns).
    eps : float, optional
        Small constant for numerical stability.

    Returns
    -------
    numpy.ndarray
        Flattened matrix of the same shape as input, containing the row-wise
        normalized results.
    """
    x = np.ascontiguousarray(input, dtype=np.float32).ravel()
    g = np.ascontiguousarray(gamma, dtype=np.float32).ravel()
    b = np.ascontiguousarray(beta, dtype=np.float32).ravel()
    row_size = int(row_size)

    output = np.empty_like(x)
    if x.size == 0:
        return output
    row_count = x.size // row_size

    d_input = cuda.mem_alloc(x.nbytes)
    d_output = cuda.mem_alloc(x.nbytes)
    d_gamma = cuda.mem_alloc(g.nbytes)
    d_beta = cuda.mem_alloc(b.nbytes)
    try:
        cuda.memcpy_htod(d_input, x)
        cuda.memcpy_htod(d_gamma, g)
        cuda.memcpy_htod(d_beta, b)

        _get_kernel()(d_input, d_gamma, d_beta, d_output,
                      np.int32(row_size), np.float32(eps),
                      block=(_THREADS, 1, 1), grid=(row_count, 1))

        cuda.memcpy_dtoh(output, d_output)
    finally:
        d_input.free()
        d_output.free()
        d_gamma.free()
        d_beta.free()

    return output
