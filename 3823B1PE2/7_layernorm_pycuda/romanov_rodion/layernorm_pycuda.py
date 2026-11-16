import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule

kernel_code = """
__global__ void layer_norm_kernel(const float* __restrict__ input, 
                                  const float* __restrict__ gamma, 
                                  const float* __restrict__ beta, 
                                  float* __restrict__ output, 
                                  int row_size, 
                                  float eps) {
    int row_idx = blockIdx.x;
    int tid = threadIdx.x;
    int bdim = blockDim.x;

    const float* d_row_in = input + row_idx * row_size;
    float* d_row_out = output + row_idx * row_size;

    // Динамическая разделяемая память для редукции (суммы и дисперсии)
    extern __shared__ float sdata[];
    float* s_sum = sdata;
    float* s_var = sdata + bdim;

    // 1. Вычисление среднего значения (Mean) по строке
    float local_sum = 0.0f;
    for (int i = tid; i < row_size; i += bdim) {
        local_sum += d_row_in[i];
    }
    s_sum[tid] = local_sum;
    __syncthreads();

    for (int stride = bdim / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_sum[tid] += s_sum[tid + stride];
        }
        __syncthreads();
    }
    float mean = s_sum[0] / row_size;
    __syncthreads();

    // 2. Вычисление дисперсии (Variance) по строке
    float local_var = 0.0f;
    for (int i = tid; i < row_size; i += bdim) {
        float diff = d_row_in[i] - mean;
        local_var += diff * diff;
    }
    s_var[tid] = local_var;
    __syncthreads();

    for (int stride = bdim / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_var[tid] += s_var[tid + stride];
        }
        __syncthreads();
    }
    float variance = s_var[0] / row_size;
    __syncthreads();

    // Обратное стандартное отклонение
    float inv_std = rsqrtf(variance + eps);

    // 3. Нормализация, применение гаммы, бета и запись в результат
    for (int i = tid; i < row_size; i += bdim) {
        float normalized = (d_row_in[i] - mean) * inv_std;
        d_row_out[i] = normalized * gamma[i] + beta[i];
    }
}
"""

def layernorm_pycuda(input, gamma, beta, row_size, eps=1e-5):
    """
    Apply Layer Normalization to each row of the input matrix using PyCUDA.
    """
    input_np = np.asarray(input, dtype=np.float32)
    gamma_np = np.asarray(gamma, dtype=np.float32)
    beta_np = np.asarray(beta, dtype=np.float32)
    
    total_elements = input_np.size
    row_count = total_elements // row_size

    output_np = np.empty_like(input_np)

    d_input = cuda.mem_alloc(input_np.nbytes)
    d_gamma = cuda.mem_alloc(gamma_np.nbytes)
    d_beta = cuda.mem_alloc(beta_np.nbytes)
    d_output = cuda.mem_alloc(output_np.nbytes)

    cuda.memcpy_htod(d_input, input_np)
    cuda.memcpy_htod(d_gamma, gamma_np)
    cuda.memcpy_htod(d_beta, beta_np)

    mod = SourceModule(kernel_code)
    kernel = mod.get_function("layer_norm_kernel")

    threads_per_block = 256
    if row_size < 256:
        threads_per_block = 32
        while threads_per_block < row_size:
            threads_per_block *= 2

    shared_mem_size = 2 * threads_per_block * 4 # 4 байта на float

    kernel(
        d_input, d_gamma, d_beta, d_output,
        np.int32(row_size), np.float32(eps),
        block=(threads_per_block, 1, 1),
        grid=(row_count, 1),
        shared=shared_mem_size
    )

    cuda.memcpy_dtoh(output_np, d_output)

    return output_np