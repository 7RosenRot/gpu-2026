#include "softmax_cuda.h"
#include <cuda_runtime.h>
#include <math_constants.h>
#include <algorithm>
#include <cstddef>

namespace {

constexpr int THREADS = 256;
constexpr int WARPS = THREADS / 32;
constexpr int STREAM_COUNT = 4;
constexpr int CACHE_SIZE = 16;

// Примерно 16 МиБ данных на одну часть.
constexpr std::size_t CHUNK_ELEMENTS = std::size_t(1) << 22;

__device__ float Reduce(
    float value,
    float* shared,
    bool maximum
) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;

    #pragma unroll
    for (int step = 16; step > 0; step >>= 1) {
        const float other = __shfl_down_sync(0xffffffffu, value, step);

        value = maximum ? fmaxf(value, other) : value + other;
    }

    if (lane == 0) {
        shared[warp] = value;
    }

    __syncthreads();

    if (warp == 0) {
        const float neutral = maximum ? -CUDART_INF_F : 0.0f;

        value = lane < WARPS ? shared[lane] : neutral;

        #pragma unroll
        for (int step = 16; step > 0; step >>= 1) {
            const float other = __shfl_down_sync(0xffffffffu, value, step);

            value = maximum ? fmaxf(value, other) : value + other;
        }

        if (lane == 0) {
            shared[0] = value;
        }
    }

    __syncthreads();

    const float result = shared[0];
    __syncthreads();
    return result;
}


__global__ void SoftmaxCached(
    float* data,
    std::size_t width
) {
    __shared__ float shared[WARPS];

    float4* row = reinterpret_cast<float4*>(data + static_cast<std::size_t>(blockIdx.x) * width);

    const std::size_t vector_count = width / 4;
    const int slots = static_cast<int>((vector_count + THREADS - 1) / THREADS);
    float4 values[CACHE_SIZE];
    float maximum = -CUDART_INF_F;

    #pragma unroll
    for (int i = 0; i < CACHE_SIZE; ++i) {
        if (i < slots) {
            const std::size_t index = threadIdx.x + i * THREADS;

            float4 v = make_float4(-CUDART_INF_F, -CUDART_INF_F, -CUDART_INF_F, -CUDART_INF_F);

            if (index < vector_count) {
                v = row[index];
            }

            values[i] = v;
            maximum = fmaxf(maximum, fmaxf(v.x, v.y));
            maximum = fmaxf(maximum, fmaxf(v.z, v.w));
        }
    }

    maximum = Reduce(maximum, shared, true);

    float sum = 0.0f;

    #pragma unroll
    for (int i = 0; i < CACHE_SIZE; ++i) {
        if (i < slots) {
            values[i].x = __expf(values[i].x - maximum);
            values[i].y = __expf(values[i].y - maximum);
            values[i].z = __expf(values[i].z - maximum);
            values[i].w = __expf(values[i].w - maximum);
            sum +=(values[i].x + values[i].y) + (values[i].z + values[i].w);
        }
    }

    const float inverse = 1.0f / Reduce(sum, shared, false);

    #pragma unroll
    for (int i = 0; i < CACHE_SIZE; ++i) {
        if (i < slots) {
            const std::size_t index = threadIdx.x + i * THREADS;

            if (index < vector_count) {
                row[index] = make_float4(
                    values[i].x * inverse,
                    values[i].y * inverse,
                    values[i].z * inverse,
                    values[i].w * inverse
                );
            }
        }
    }
}

// Универсальный вариант для остальных размеров строк.
__global__ void SoftmaxGeneric(float* data, std::size_t width) {
    __shared__ float shared[WARPS];

    float* row = data + static_cast<std::size_t>(blockIdx.x) * width;

    float maximum = -CUDART_INF_F;

    for (std::size_t col = threadIdx.x; col < width; col += THREADS) {
        maximum = fmaxf(maximum, row[col]);
    }

    maximum = Reduce(maximum, shared, true);

    float sum = 0.0f;

    for (std::size_t col = threadIdx.x; col < width; col += THREADS) {
        sum += __expf(row[col] - maximum);
    }
    const float inverse = 1.0f / Reduce(sum, shared, false);
    for (std::size_t col = threadIdx.x; col < width; col += THREADS) {
        row[col] = __expf(row[col] - maximum) * inverse;
    }
}

void LaunchSoftmax(float* data, std::size_t rows, std::size_t width, cudaStream_t stream) {
    const unsigned int blocks = static_cast<unsigned int>(rows);

    if (width % 4 == 0 &&
        width <= static_cast<std::size_t>(THREADS) * CACHE_SIZE * 4) {
        SoftmaxCached<<<blocks, THREADS, 0, stream>>>(data, width);
    } else {
        SoftmaxGeneric<<<blocks, THREADS, 0, stream>>>(data, width);
    }
}

} // namespace

std::vector<float> SoftmaxCUDA(const std::vector<float>& input,int row_count) {
    if (row_count <= 0 || input.empty()) {
        return {};
    }

    const std::size_t rows = static_cast<std::size_t>(row_count);
    const std::size_t count = input.size();
    if (count % rows != 0) {
        return {};
    }

    const std::size_t width = count / rows;
    const std::size_t bytes = count * sizeof(float);
    std::vector<float> output(count);

    if (count <= CHUNK_ELEMENTS) {
        float* device = nullptr;

        cudaMalloc(reinterpret_cast<void**>(&device), bytes);
        cudaMemcpy(device, input.data(), bytes, cudaMemcpyHostToDevice);
        LaunchSoftmax(device, rows, width, nullptr);
        cudaMemcpy(output.data(), device, bytes, cudaMemcpyDeviceToHost);
        cudaFree(device);


        return output;
    }


    const std::size_t rows_per_chunk = std::min(rows, std::max(std::size_t(1), CHUNK_ELEMENTS / width));
    const std::size_t chunk_elements = rows_per_chunk * width;
    const std::size_t chunk_bytes = chunk_elements * sizeof(float);
    cudaStream_t streams[STREAM_COUNT] = {};
    float* buffers[STREAM_COUNT] = {};
    for (int i = 0; i < STREAM_COUNT; ++i) {
        cudaStreamCreateWithFlags(&streams[i], cudaStreamNonBlocking);
        cudaMalloc(reinterpret_cast<void**>(&buffers[i]), chunk_bytes);
    }


    const bool input_registered = cudaHostRegister(const_cast<float*>(input.data()), bytes, cudaHostRegisterDefault) == cudaSuccess;
    const bool output_registered = cudaHostRegister(output.data(), bytes, cudaHostRegisterDefault) == cudaSuccess;
    std::size_t chunk_index = 0;

    for (std::size_t first_row = 0; first_row < rows; first_row += rows_per_chunk, ++chunk_index) {
        const int slot = static_cast<int>(chunk_index % STREAM_COUNT);
        const std::size_t current_rows = std::min(rows_per_chunk, rows - first_row);
        const std::size_t offset = first_row * width;
        const std::size_t current_bytes = current_rows * width * sizeof(float);
        cudaStream_t stream = streams[slot];
        float* device = buffers[slot];

        cudaMemcpyAsync(device, input.data() + offset, current_bytes, cudaMemcpyHostToDevice, stream);
        LaunchSoftmax(device, current_rows, width, stream);
        cudaMemcpyAsync(output.data() + offset, device, current_bytes, cudaMemcpyDeviceToHost, stream);
    }

    for (int i = 0; i < STREAM_COUNT; ++i) {
        cudaStreamSynchronize(streams[i]);
    }
    if (input_registered) {
        cudaHostUnregister(const_cast<float*>(input.data()));
    }
    if (output_registered) {
        cudaHostUnregister(output.data());
    }
    for (int i = 0; i < STREAM_COUNT; ++i) {
        cudaFree(buffers[i]);
        cudaStreamDestroy(streams[i]);
    }



    return output;
}