#include "gelu_cuda.h"
#include <cmath>
#include <cuda_runtime.h>

// Ядро (kernel) — это функция, которая выполняется на GPU.
// __global__ означает, что функция вызывается с CPU, а выполняется на GPU.
__global__ void gelu_kernel(const float* __restrict__ input, 
                            float* __restrict__ output, 
                            int n) {
    // Вычисляем глобальный индекс потока
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Проверяем, не вышли ли мы за границы массива
    if (i < n) {
        float x = input[i];
        
        // Константы для быстрой формулы GELU (через exp вместо tanh)
        const float kSqrt2OverPi = 0.7978845608028654f;
        const float kA = -2.0f * kSqrt2OverPi;
        const float kB = kA * 0.044715f;

        // Вычисляем t = -2 * sqrt(2/pi) * (x + 0.044715 * x^3)
        float t = x * (kA + kB * x * x);
        
        // Финальная формула GELU(x) = x / (1 + exp(t))
        output[i] = x / (1.0f + std::exp(t));
    }
}

std::vector<float> GeluCUDA(const std::vector<float>& input) {
    int n = input.size();
    std::vector<float> output(n);

    // Указатели для памяти GPU
    float* d_input = nullptr;
    float* d_output = nullptr;

    // Выделяем память на видеокарте (device)
    cudaMalloc(&d_input, n * sizeof(float));
    cudaMalloc(&d_output, n * sizeof(float));

    // Копируем данные с CPU (host) на GPU (device)
    cudaMemcpy(d_input, input.data(), n * sizeof(float), cudaMemcpyHostToDevice);

    // Настраиваем параметры запуска ядра
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock; // Округление вверх

    // Запускаем ядро на GPU
    gelu_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, n);

    // Ждем, пока GPU закончит вычисления, и копируем результат обратно на CPU
    cudaMemcpy(output.data(), d_output, n * sizeof(float), cudaMemcpyDeviceToHost);

    // Освобождаем память GPU
    cudaFree(d_input);
    cudaFree(d_output);

    return output;
}