#include <KernelTutorial/Gemm.hpp>

// Basic GEMM
void Gemm(float* a, float* b, float* c, float alpha, float beta, std::size_t m, std::size_t n, std::size_t k)
{
    for(std::size_t mIdx = 0; mIdx < m; ++mIdx)
        for(std::size_t nIdx = 0; nIdx < n; ++nIdx)
        {
            float sum = 0;

            for(std::size_t kIdx = 0; kIdx < k; ++kIdx)
            {
                sum += a[(k * mIdx) + kIdx] * b[(n * kIdx) + nIdx];
            }

            c[(n * mIdx) + nIdx] = (alpha * sum) + (beta * c[(n * mIdx) + nIdx]);
        }
}