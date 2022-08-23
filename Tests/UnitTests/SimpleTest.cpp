#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include <KernelTutorial/Test.hpp>
#include <KernelTutorial/Gemm.hpp>
#include <blaze/Blaze.h>
#include <random>
#include <memory>
#include <chrono>
#include <cstdlib>

std::chrono::microseconds testGemm(GemmKernel kernel, std::size_t m, std::size_t n, std::size_t k, float alpha, float beta)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    // std::uniform_real_distribution dist(-1.0f, 0.0f);
    const auto dist = [](std::mt19937&) -> float { return 1.0; };
    //std::unique_ptr<float[]> arrA(reinterpret_cast<float*>(std::aligned_alloc(16, sizeof(float) * m * k)) /*new float[m*k]*/);
    //std::unique_ptr<float[]> arrB(reinterpret_cast<float*>(std::aligned_alloc(16, sizeof(float) * k * n)) /*new float[k*n]*/);
    //std::unique_ptr<float[]> arrC(reinterpret_cast<float*>(std::aligned_alloc(16, sizeof(float) * m * n)) /*new float[m*n]*/);
    float* arrA =
        reinterpret_cast<float*>(_aligned_malloc(sizeof(float) * m * k, 64));
    float* arrB =
        reinterpret_cast<float*>(_aligned_malloc(sizeof(float) * k * m, 64));
    float* arrC =
        reinterpret_cast<float*>(_aligned_malloc(sizeof(float) * m * n, 64));

    for(std::size_t i = 0; i < m*k; ++i)
        arrA[i] = dist(gen);
    for(std::size_t i = 0; i < k*n; ++i)
        arrB[i] = dist(gen);
    for(std::size_t i = 0; i < m*n; ++i)
        arrC[i] = dist(gen);

    blaze::DynamicMatrix<float, blaze::rowMajor> a(m, k, 0.0f);
    blaze::DynamicMatrix<float, blaze::rowMajor> b(k, n, 0.0f);
    blaze::DynamicMatrix<float, blaze::rowMajor> c(m, n, 0.0f);

    for(std::size_t i = 0; i < m; ++i)
        for(std::size_t j = 0; j < k; ++j)
            a(i, j) = arrA[i*k + j];

    for(std::size_t i = 0; i < k; ++i)
        for(std::size_t j = 0; j < n; ++j)
            b(i, j) = arrB[i*n + j];

    for(std::size_t i = 0; i < m; ++i)
        for(std::size_t j = 0; j < n; ++j)
            c(i, j) = arrC[i*n + j];

    const auto t1 = std::chrono::system_clock::now();
    // kernel(arrA.get(), arrB.get(), arrC.get(), alpha, beta, m, n, k);
    kernel(arrA, arrB, arrC, alpha, beta, m, n, k);
    const auto t2 = std::chrono::system_clock::now();

    const auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1);

    c = alpha*a*b + beta*c;
    for(std::size_t i = 0; i < m; ++i)
        for(std::size_t j = 0; j < n; ++j)
        {
            if (std::abs(c(i, j) - arrC[i*n + j]) > std::numeric_limits<float>::epsilon() * 10000)
            {
                std::cout << c(i, j) << " vs " << arrC[i * n + j] << std::endl;
            }
        }

    return elapsed;
}

TEST_CASE("parallel vs parallel_tiled")
{
    for (auto i = 0; i < 10; i++)
    {
        auto elapsedP = testGemm(GemmParallel, 200, 200, 200, 1.0, 1.0);
        auto elapsedPT = testGemm(GemmParallelTiled, 200, 200, 200, 1.0, 1.0);

        // std::cout << "#" << i << " Parallel: " << elapsedP.count() << "ms vs. ParallelTiled: " << elapsedPT.count() << "ms" << std::endl;
    }
}

TEST_CASE("tiled vs parallel vs unrolled")
{
    constexpr std::size_t N = 256;
    
    for (auto i = 0; i < 10; i++)
    {
        auto elapsedB = testGemm(GemmBasic, N, N, N, 1.0, 1.0);
        auto elapsedT = testGemm(GemmTiling, N, N, N, 1.0, 1.0);
        auto elapsedP = testGemm(GemmParallelTiled, N, N, N, 1.0, 1.0);
        auto elapsedU = testGemm(GemmUnrolled, N, N, N, 1.0, 1.0);
        auto elapsedV = testGemm(GemmVectorized, N, N, N, 1.0, 1.0);

        std::cout << "#" << i << " Basic: " << elapsedB.count()
                  << "ms vs. Tiled: " << elapsedT.count()
                  << "ms vs. Parallel: " << elapsedP.count()
                  << "ms vs. Unrolled: " << elapsedU.count()
                  << "ms vs. Vectorized: " << elapsedV.count() << "ms"
                  << std::endl;
    }
}