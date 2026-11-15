import numpy as np


_KERNEL_SOURCE = r"""
extern "C" __global__ void layernorm_kernel(
    const float* input,
    const float* gamma,
    const float* beta,
    float* output,
    int row_size,
    float eps) {
    extern __shared__ float reduction[];

    const unsigned int tid = threadIdx.x;
    const size_t row_start = static_cast<size_t>(blockIdx.x) * row_size;
    const float shift = input[row_start];

    // Shift before summing to improve accuracy for rows with a large offset.
    float partial_sum = 0.0f;
    for (int col = tid; col < row_size; col += blockDim.x) {
        partial_sum += input[row_start + col] - shift;
    }
    reduction[tid] = partial_sum;
    __syncthreads();

    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            reduction[tid] += reduction[tid + stride];
        }
        __syncthreads();
    }
    const float mean = shift + reduction[0] / static_cast<float>(row_size);

    float partial_variance = 0.0f;
    for (int col = tid; col < row_size; col += blockDim.x) {
        const float difference = input[row_start + col] - mean;
        partial_variance += difference * difference;
    }
    reduction[tid] = partial_variance;
    __syncthreads();

    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            reduction[tid] += reduction[tid + stride];
        }
        __syncthreads();
    }
    const float variance = reduction[0] / static_cast<float>(row_size);
    const float inverse_stddev = rsqrtf(variance + eps);

    for (int col = tid; col < row_size; col += blockDim.x) {
        const float normalized = (input[row_start + col] - mean) * inverse_stddev;
        output[row_start + col] = normalized * gamma[col] + beta[col];
    }
}
"""

_cached_module = None
_cached_kernel = None
_cached_context = None
_cached_buffers = [None, None, None, None]
_cached_buffer_sizes = [0, 0, 0, 0]


def _get_kernel(cuda, compiler):
    global _cached_module, _cached_kernel, _cached_context
    global _cached_buffers, _cached_buffer_sizes

    context = cuda.Context.get_current()
    if _cached_kernel is None or context != _cached_context:
        _cached_module = compiler.SourceModule(_KERNEL_SOURCE, options=["-O3"])
        _cached_kernel = _cached_module.get_function("layernorm_kernel")
        _cached_context = context
        _cached_buffers = [None, None, None, None]
        _cached_buffer_sizes = [0, 0, 0, 0]
    return _cached_kernel


def layernorm_pycuda(input, gamma, beta, row_size, eps=1e-5):
    """Apply LayerNorm independently to each row of a flattened matrix on CUDA."""
    try:
        parsed_row_size = int(row_size)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("row_size must be a positive integer") from exc
    if parsed_row_size <= 0 or parsed_row_size != row_size:
        raise ValueError("row_size must be a positive integer")
    row_size = parsed_row_size

    try:
        eps = float(eps)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("eps must be a non-negative number") from exc
    if not np.isfinite(eps) or eps < 0.0:
        raise ValueError("eps must be a finite, non-negative number")

    values = np.ascontiguousarray(input, dtype=np.float32).reshape(-1)
    scale = np.ascontiguousarray(gamma, dtype=np.float32).reshape(-1)
    bias = np.ascontiguousarray(beta, dtype=np.float32).reshape(-1)

    if values.size % row_size != 0:
        raise ValueError("input length must be divisible by row_size")
    if scale.size != row_size or bias.size != row_size:
        raise ValueError("gamma and beta must each have length row_size")
    if values.size == 0:
        return np.empty_like(values)

    # Import lazily so the module can be imported on a machine without CUDA.
    import pycuda.autoinit  # noqa: F401
    import pycuda.driver as cuda
    import pycuda.compiler as compiler

    row_count = values.size // row_size
    output = np.empty_like(values)

    kernel = _get_kernel(cuda, compiler)
    required_sizes = [values.nbytes, scale.nbytes, bias.nbytes, output.nbytes]
    for index, required_size in enumerate(required_sizes):
        if required_size > _cached_buffer_sizes[index]:
            _cached_buffers[index] = cuda.mem_alloc(required_size)
            _cached_buffer_sizes[index] = required_size

    device_input, device_gamma, device_beta, device_output = _cached_buffers

    cuda.memcpy_htod(device_input, values)
    cuda.memcpy_htod(device_gamma, scale)
    cuda.memcpy_htod(device_beta, bias)

    threads = 1
    while threads * 2 <= min(row_size, 256):
        threads *= 2

    kernel(
        device_input,
        device_gamma,
        device_beta,
        device_output,
        np.int32(row_size),
        np.float32(eps),
        block=(threads, 1, 1),
        grid=(row_count, 1, 1),
        shared=threads * np.dtype(np.float32).itemsize,
    )

    cuda.memcpy_dtoh(output, device_output)
    return output
