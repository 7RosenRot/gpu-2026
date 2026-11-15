import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule

_MODULE = SourceModule("""
__global__ void layernorm_kernel(const float* __restrict__ input,
                                 const float* __restrict__ gamma,
                                 const float* __restrict__ beta,
                                 float* __restrict__ output,
                                 int row_size,
                                 float eps) {
    extern __shared__ float sdata[];
    int row = blockIdx.x;
    int tid = threadIdx.x;
    const float* row_in = input + row * row_size;
    float* row_out = output + row * row_size;

    float sum = 0.0f;
    for (int i = tid; i < row_size; i += blockDim.x) {
        sum += row_in[i];
    }
    sdata[tid] = sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    float mean = sdata[0] / row_size;
    __syncthreads();

    float var_sum = 0.0f;
    for (int i = tid; i < row_size; i += blockDim.x) {
        float diff = row_in[i] - mean;
        var_sum += diff * diff;
    }
    sdata[tid] = var_sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    float inv_std = rsqrtf(sdata[0] / row_size + eps);

    for (int i = tid; i < row_size; i += blockDim.x) {
        row_out[i] = (row_in[i] - mean) * inv_std * gamma[i] + beta[i];
    }
}
""")

_kernel = _MODULE.get_function("layernorm_kernel")


def layernorm_pycuda(input, gamma, beta, row_size, eps=1e-5):
    input = np.asarray(input, dtype=np.float32)
    gamma = np.asarray(gamma, dtype=np.float32)
    beta = np.asarray(beta, dtype=np.float32)

    row_count = input.size // row_size
    output = np.empty_like(input)

    d_input = cuda.mem_alloc(input.nbytes)
    d_gamma = cuda.mem_alloc(gamma.nbytes)
    d_beta = cuda.mem_alloc(beta.nbytes)
    d_output = cuda.mem_alloc(output.nbytes)

    cuda.memcpy_htod(d_input, input)
    cuda.memcpy_htod(d_gamma, gamma)
    cuda.memcpy_htod(d_beta, beta)

    threads = 256
    blocks = row_count
    shared_mem = threads * np.dtype(np.float32).itemsize

    _kernel(d_input, d_gamma, d_beta, d_output,
            np.int32(row_size), np.float32(eps),
            block=(threads, 1, 1), grid=(blocks, 1, 1),
            shared=shared_mem)

    cuda.memcpy_dtoh(output, d_output)
    return output