#include "gelu_cuda.h"

#include <cuda_runtime.h>
#include <sys/mman.h>

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <thread>

namespace {

// Проверка ошибок CUDA (лекция 3, слайд 6): без неё ошибка на GPU проходит
// молча, и мы просто получаем неверный результат.
void Check(cudaError_t error) {
    if (error != cudaSuccess) {
        std::fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(error));
        std::exit(1);
    }
}

// Та же формула, что в лабе 1: 0.5 * (1 + tanh(u)) = 1 / (1 + e^(-2u)),
// поэтому GELU(x) = x / (1 + e^(-2u)), где 2u = x * (kC0 + kC1 * x^2).
constexpr float kC0 = 1.5957691216057308f;  // 2 * sqrt(2 / pi)
constexpr float kC1 = kC0 * 0.044715f;      // 2 * sqrt(2 / pi) * 0.044715
constexpr int kThreadsPerBlock = 256;

// Ядро: каждый поток GPU обрабатывает один элемент, прямо на месте (in-place),
// поэтому на видеокарте нужен один буфер, а не два.
// __expf — быстрая встроенная экспонента GPU (подсказка «better formula»).
// Своя экспонента, как в лабе 1, здесь не нужна.
// При очень больших отрицательных x: __expf(+inf) = inf, и x / inf = -0 — верный ответ.
__global__ void GeluKernel(float* data, std::size_t n) {
    const std::size_t i = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
    if (i < n) {
        const float x = data[i];
        data[i] = x / (1.0f + __expf(-x * (kC0 + kC1 * x * x)));
    }
}

// Подсказка «allocate and free device memory once»: cudaMalloc медленный,
// а функцию вызывают много раз. Поэтому буфер на GPU выделяем при первом
// вызове и дальше используем повторно; новый — только если нужен больший.
// Освобождать вручную не нужно: драйвер освободит его при выходе из программы.
float* DeviceBuffer(std::size_t n) {
    static float* buffer = nullptr;
    static std::size_t capacity = 0;
    if (n > capacity) {
        if (buffer != nullptr) {
            Check(cudaFree(buffer));
        }
        Check(cudaMalloc(&buffer, n * sizeof(float)));
        capacity = n;
    }
    return buffer;
}

// Как в лабе 1: просим Linux выделять память страницами по 2 МБ вместо 4 КБ,
// чтобы обращений к ОС за новыми страницами было в 512 раз меньше.
void UseHugePages(void* ptr, std::size_t bytes) {
    constexpr std::uintptr_t kHuge = std::uintptr_t{2} << 20;
    const auto addr = reinterpret_cast<std::uintptr_t>(ptr);
    const std::uintptr_t begin = (addr + kHuge - 1) & ~(kHuge - 1);
    const std::uintptr_t end = (addr + bytes) & ~(kHuge - 1);
    if (end > begin) {
        madvise(reinterpret_cast<void*>(begin), end - begin, MADV_HUGEPAGE);
    }
}

// Готовит вектор под результат: выделяет и заполняет нулями 512 МБ.
// Это занимает десятки миллисекунд, поэтому делаем это в отдельном потоке CPU.
void PrepareOutput(std::vector<float>* output, std::size_t n) {
    output->reserve(n);
    UseHugePages(output->data(), n * sizeof(float));
    output->resize(n);
}

}  // namespace

std::vector<float> GeluCUDA(const std::vector<float>& input) {
    const std::size_t n = input.size();
    std::vector<float> output;
    if (n == 0) {
        return output;
    }
    const std::size_t bytes = n * sizeof(float);
    float* d_data = DeviceBuffer(n);

    // Подсказка «overlap host memory allocation and CUDA computations»:
    // пока основной поток копирует данные в GPU и запускает ядро,
    // второй поток CPU параллельно готовит память под результат.
    std::thread preparer(PrepareOutput, &output, n);

    // Копируем вход в память видеокарты (CPU -> GPU).
    Check(cudaMemcpy(d_data, input.data(), bytes, cudaMemcpyHostToDevice));

    // Запускаем ядро: блоки по 256 потоков, блоков столько, чтобы хватило на все n.
    const unsigned blocks = static_cast<unsigned>((n + kThreadsPerBlock - 1) / kThreadsPerBlock);
    GeluKernel<<<blocks, kThreadsPerBlock>>>(d_data, n);
    Check(cudaGetLastError());  // ошибка запуска ядра (например, неверные размеры)

    // Ждём, пока память под результат готова, и копируем результат обратно (GPU -> CPU).
    // cudaMemcpy сам дождётся окончания ядра: они в одной очереди (stream 0).
    preparer.join();
    Check(cudaMemcpy(output.data(), d_data, bytes, cudaMemcpyDeviceToHost));

    return output;
}
