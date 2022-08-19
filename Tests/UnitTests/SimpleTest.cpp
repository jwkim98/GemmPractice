#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include <KernelTutorial/Test.hpp>
#include <KernelTutorial/Gemm.hpp>
#include <blaze/Blaze.h>
#include <random>
#include <memory>
#include <chrono>


void testGemm(std::size_t m, std::size_t n, std::size_t k, float alpha, float beta)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution dist(-1.0f, 0.0f);
    std::unique_ptr<float[]> arrA(new float[m*k]);
    std::unique_ptr<float[]> arrB(new float[k*n]);
    std::unique_ptr<float[]> arrC(new float[m*n]);

    for(std::size_t i = 0; i < m*k; ++i)
        arrA[i] = dist(gen);
    for(std::size_t i = 0; i < k*n; ++i)
        arrB[i] = dist(gen);
    for(std::size_t i = 0; i < m*n; ++i)
    {
                arrC[i] = dist(gen);
    }

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
    Gemm(arrA.get(), arrB.get(), arrC.get(), alpha, beta, m, n, k);
    const auto t2 = std::chrono::system_clock::now();

    std::cout<<"elapsed time : "<<std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count()<<std::endl;

    c = alpha*a*b + beta*c;
    for(std::size_t i = 0; i < m; ++i)
        for(std::size_t j = 0; j < n; ++j)
        {
            CHECK(std::abs(c(i, j) - arrC[i*n + j]) < std::numeric_limits<float>::epsilon());
        }
}

TEST_CASE("Gemm")
{
    std::random_device rd;
    std::default_random_engine engine(rd());
    std::uniform_int_distribution<std::size_t> intDist(100, 200);
    std::uniform_real_distribution<float> realDist(-5.0, 5.0);

    for (auto i = 0; i < 10; ++i)
    {
        std::size_t M = intDist(engine);
        std::size_t N = intDist(engine);
        std::size_t K = intDist(engine);
        float alpha = realDist(engine);
        float beta = realDist(engine);

        std::cout << "#" << i << " - M: " << M << ", N: " << N << ", K: " << K << ", alpha: " << alpha << ", beta: " << beta << std::endl;

        testGemm(M, N, K, alpha, beta);
    }
}