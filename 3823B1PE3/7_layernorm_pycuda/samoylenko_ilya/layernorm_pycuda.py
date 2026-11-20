import atexit
import operator

import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
from pycuda.tools import DeviceMemoryPool, PageLockedMemoryPool


_module = SourceModule(r"""
__device__ double block_sum(double value, double* shared_vals) {
    int thread = threadIdx.x;
    shared_vals[thread] = value;
    __syncthreads();

    for (int step = blockDim.x / 2; step > 0; step /= 2) {
        if (thread < step) {
            shared_vals[thread] += shared_vals[thread + step];
        }
        __syncthreads();
    }
    return shared_vals[0];
}

__global__ void LayerNormKernel(float* data, const float* gamma,
                                const float* beta, int row_size, float eps) {
    __shared__ double shared_vals[256];
    int thread = threadIdx.x;
    float* row = data + static_cast<size_t>(blockIdx.x) * row_size;

    double sum = 0.0;
    for (int col = thread; col < row_size; col += blockDim.x) {
        sum += row[col];
    }

    const double mean = block_sum(sum, shared_vals) / row_size;
    __syncthreads();

    sum = 0.0;
    for (int col = thread; col < row_size; col += blockDim.x) {
        double difference = row[col] - mean;
        sum += difference * difference;
    }

    const double row_var = block_sum(sum, shared_vals) / row_size;
    const double inv_sqrt_var = rsqrt(row_var + eps);

    for (int col = thread; col < row_size; col += blockDim.x) {
        row[col] = (row[col] - mean) * inv_sqrt_var * gamma[col] + beta[col];
    }
}
""")
_kernel = _module.get_function("LayerNormKernel")
_device_pool = DeviceMemoryPool()
_output_pool = PageLockedMemoryPool()
atexit.register(_output_pool.stop_holding)
atexit.register(_device_pool.stop_holding)


def layernorm_pycuda(input, gamma, beta, row_size, eps=1e-5):
    row_size = operator.index(row_size)
    data = np.ascontiguousarray(input, dtype=np.float32).reshape(-1)
    gamma = np.ascontiguousarray(gamma, dtype=np.float32)
    beta = np.ascontiguousarray(beta, dtype=np.float32)

    if row_size <= 0 or row_size & (row_size - 1):
        raise ValueError("row_size should be 2^n")
    if data.size % row_size or gamma.size != row_size or beta.size != row_size:
        raise ValueError("input, gamma, beta len should = row_size")
    if data.size == 0:
        return np.empty_like(data)

    try:
        output = _output_pool.allocate(data.shape, data.dtype)
    except cuda.MemoryError:
        output = np.empty_like(data)

    device = _device_pool.allocate(data.nbytes)
    cuda.memcpy_htod(device, data)

    _kernel(
        device, cuda.In(gamma), cuda.In(beta),
        np.int32(row_size), np.float32(eps),
        block=(min(row_size, 256), 1, 1),
        grid=(data.size // row_size, 1, 1),
    )

    cuda.memcpy_dtoh(output, device)
    return output
