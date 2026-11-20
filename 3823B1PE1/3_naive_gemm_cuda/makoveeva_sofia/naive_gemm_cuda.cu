#include "naive_gemm_cuda.h"

#include <cuda_runtime.h>
#include <sys/mman.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <thread>

namespace {

void Check(cudaError_t error) {
    if (error != cudaSuccess) {
        std::fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(error));
        std::exit(1);
    }
}

// ---------------------------------------------------------------------------
// Ядра
// ---------------------------------------------------------------------------

// Простое ядро: один поток = один элемент C. Используется только для
// маленьких матриц (n < 128), которые не делятся на плитки основного ядра.
// threadIdx.x -> столбец: соседние потоки warp-а читают соседние элементы
// строки B (одна общая транзакция), а a[row][k] у них одинаковый (рассылка).
__global__ void SimpleGemmKernel(const float* a, const float* b, float* c, int n) {
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < n && col < n) {
        float sum = 0.0f;
        for (int k = 0; k < n; ++k) {
            sum += a[row * n + k] * b[k * n + col];
        }
        c[row * n + col] = sum;
    }
}

// Основное ядро. Каждый поток считает плитку C размером kRows x 4:
// kRows строк и 4 соседних столбца (подсказка «multiple elements per thread»).
//
// Почему это быстрее: в простом ядре на каждое умножение приходится 2 чтения
// из памяти. Здесь за один шаг (4 значения k) поток читает 4 float4 из B
// и kRows float4 из A, а делает 16 * kRows умножений: одно прочитанное
// число используется много раз, пока лежит в регистре.
//
// float4 — 4 числа одной инструкцией чтения (подсказка «load vectorization»).
// Внутренние циклы с #pragma unroll компилятор разворачивает полностью
// (подсказка «loop unrolling»), и acc[], b_rows[] живут в регистрах.
constexpr int kBlockX = 32;  // потоков блока по столбцам = ровно один warp
constexpr int kBlockY = 8;   // потоков блока по строкам
constexpr int kRows = 4;     // строк C на один поток
constexpr int kTileCols = kBlockX * 4;      // 128 столбцов C на блок
constexpr int kTileRows = kBlockY * kRows;  // 32 строки C на блок

__device__ __forceinline__ float4 Load4(const float* m, int row, int col, int n) {
    return *reinterpret_cast<const float4*>(m + static_cast<std::size_t>(row) * n + col);
}

// acc += s * v для всех четырёх компонент (fmaf = умножение и сложение за одну операцию).
__device__ __forceinline__ float4 Fma4(float s, float4 v, float4 acc) {
    return make_float4(fmaf(s, v.x, acc.x), fmaf(s, v.y, acc.y),
                       fmaf(s, v.z, acc.z), fmaf(s, v.w, acc.w));
}

__global__ void TiledGemmKernel(const float* __restrict__ a, const float* __restrict__ b,
                                float* __restrict__ c, int n) {
    const int col = 4 * (blockIdx.x * kBlockX + threadIdx.x);      // первый из 4 столбцов
    const int row = kRows * (blockIdx.y * kBlockY + threadIdx.y);  // первая из kRows строк

    float4 acc[kRows];
#pragma unroll
    for (int r = 0; r < kRows; ++r) {
        acc[r] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    }

    for (int k = 0; k < n; k += 4) {
        // B[k..k+3][col..col+3]. Warp (32 потока подряд по x) читает 32 float4
        // подряд = 512 байт одним куском — это и есть «warp-friendly» доступ.
        float4 b_rows[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            b_rows[i] = Load4(b, k + i, col, n);
        }
#pragma unroll
        for (int r = 0; r < kRows; ++r) {
            // A[row+r][k..k+3]: у всех потоков warp-а один и тот же адрес,
            // память читается один раз и рассылается всем 32 потокам.
            const float4 a4 = Load4(a, row + r, k, n);
            acc[r] = Fma4(a4.x, b_rows[0], acc[r]);
            acc[r] = Fma4(a4.y, b_rows[1], acc[r]);
            acc[r] = Fma4(a4.z, b_rows[2], acc[r]);
            acc[r] = Fma4(a4.w, b_rows[3], acc[r]);
        }
    }

#pragma unroll
    for (int r = 0; r < kRows; ++r) {
        *reinterpret_cast<float4*>(c + static_cast<std::size_t>(row + r) * n + col) = acc[r];
    }
}

// ---------------------------------------------------------------------------
// Память и потоки CUDA — создаются один раз и переиспользуются (как в лабе 2)
// ---------------------------------------------------------------------------

constexpr int kMaxChunks = 8;  // на сколько горизонтальных полос делим A и C

struct GpuContext {
    float* a = nullptr;
    float* b = nullptr;
    float* c = nullptr;
    std::size_t capacity = 0;
    cudaStream_t streams[kMaxChunks];

    GpuContext() {
        for (cudaStream_t& s : streams) {
            Check(cudaStreamCreate(&s));
        }
    }

    void Reserve(std::size_t count) {
        if (count <= capacity) {
            return;
        }
        if (a != nullptr) {
            Check(cudaFree(a));
            Check(cudaFree(b));
            Check(cudaFree(c));
        }
        Check(cudaMalloc(&a, count * sizeof(float)));
        Check(cudaMalloc(&b, count * sizeof(float)));
        Check(cudaMalloc(&c, count * sizeof(float)));
        capacity = count;
    }
};

GpuContext& Gpu(std::size_t count) {
    static GpuContext context;  // создаётся при первом вызове
    context.Reserve(count);
    return context;
}

// Как в лабах 1-2: huge pages + заполнение вектора в отдельном потоке CPU.
void PrepareOutput(std::vector<float>* output, std::size_t count) {
    output->reserve(count);
    constexpr std::uintptr_t kHuge = std::uintptr_t{2} << 20;
    const auto addr = reinterpret_cast<std::uintptr_t>(output->data());
    const std::uintptr_t begin = (addr + kHuge - 1) & ~(kHuge - 1);
    const std::uintptr_t end = (addr + count * sizeof(float)) & ~(kHuge - 1);
    if (end > begin) {
        madvise(reinterpret_cast<void*>(begin), end - begin, MADV_HUGEPAGE);
    }
    output->resize(count);
}

}  // namespace

std::vector<float> NaiveGemmCUDA(const std::vector<float>& a,
                                 const std::vector<float>& b,
                                 int n) {
    std::vector<float> c;
    if (n <= 0) {
        return c;
    }
    const std::size_t count = static_cast<std::size_t>(n) * n;
    GpuContext& gpu = Gpu(count);

    // Подсказка «overlap host memory allocation»: память под C готовит
    // второй поток CPU, пока GPU копирует и считает.
    std::thread preparer(PrepareOutput, &c, count);

    // B нужна каждому блоку целиком — копируем её первой и ждём окончания.
    Check(cudaMemcpy(gpu.b, b.data(), count * sizeof(float), cudaMemcpyHostToDevice));

    if (n % kTileCols != 0 || n % kTileRows != 0) {
        // Маленькие матрицы (n < 128): простое ядро, без конвейера.
        Check(cudaMemcpy(gpu.a, a.data(), count * sizeof(float), cudaMemcpyHostToDevice));
        const dim3 block(16, 16);
        const dim3 grid((n + 15) / 16, (n + 15) / 16);
        SimpleGemmKernel<<<grid, block>>>(gpu.a, gpu.b, gpu.c, n);
        Check(cudaGetLastError());
        preparer.join();
        Check(cudaMemcpy(c.data(), gpu.c, count * sizeof(float), cudaMemcpyDeviceToHost));
        return c;
    }

    // Конвейер (лекция 3, потоки CUDA): режем A и C на полосы по строкам.
    // Полоса s строк C зависит только от полосы s строк A (и всей B), поэтому
    // каждую полосу ведём в своём потоке CUDA: копия A -> ядро -> копия C.
    // Пока GPU считает полосу s, по шине уже едет полоса s+1 матрицы A,
    // а результат полосы s едет обратно, пока считаются следующие.
    const int chunks = std::min(kMaxChunks, n / kTileRows);
    const int chunk_rows = n / chunks;  // n — степень двойки, делится нацело
    const std::size_t chunk_count = static_cast<std::size_t>(chunk_rows) * n;
    const dim3 block(kBlockX, kBlockY);
    const dim3 grid(n / kTileCols, chunk_rows / kTileRows);

    // Сначала ставим в очереди все копирования A и все ядра...
    for (int s = 0; s < chunks; ++s) {
        const std::size_t offset = s * chunk_count;
        Check(cudaMemcpyAsync(gpu.a + offset, a.data() + offset, chunk_count * sizeof(float),
                              cudaMemcpyHostToDevice, gpu.streams[s]));
        TiledGemmKernel<<<grid, block, 0, gpu.streams[s]>>>(gpu.a + offset, gpu.b,
                                                            gpu.c + offset, n);
        Check(cudaGetLastError());
    }

    // ...потом, когда память под C готова, забираем полосы результата.
    preparer.join();
    for (int s = 0; s < chunks; ++s) {
        const std::size_t offset = s * chunk_count;
        Check(cudaMemcpyAsync(c.data() + offset, gpu.c + offset, chunk_count * sizeof(float),
                              cudaMemcpyDeviceToHost, gpu.streams[s]));
    }
    Check(cudaDeviceSynchronize());
    return c;
}
