#include <KernelTutorial/Gemm.hpp>

#include <algorithm>

// Basic GEMM
void GemmBasic(float* a, float* b, float* c, float alpha, float beta, std::size_t m, std::size_t n, std::size_t k)
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

// GEMM with loop tiling optimization
void GemmTiling(float* a, float* b, float* c, float alpha, float beta, std::size_t m, std::size_t n, std::size_t k)
{
    for (std::size_t i = 0; i < (m*n); ++i)
        c[i] *= beta; // beta*C 부분을 미리 계산.

    constexpr std::size_t CacheSize = 64;

    for (std::size_t mBlock = 0; mBlock < m; mBlock += CacheSize)
        for (std::size_t kBlock = 0; kBlock < k; kBlock += CacheSize)
            for (std::size_t nBlock = 0; nBlock < n; nBlock += CacheSize)
            {
                const std::size_t mEnd = std::min(mBlock + CacheSize, m);
                const std::size_t nEnd = std::min(nBlock + CacheSize, n);
                const std::size_t kEnd = std::min(kBlock + CacheSize, k);

                for (std::size_t mIdx = mBlock; mIdx < mEnd; ++mIdx)
                    for (std::size_t kIdx = kBlock; kIdx < kEnd; ++kIdx)
                    {
                        for (std::size_t nIdx = nBlock; nIdx < nEnd; ++nIdx)
                            c[(n * mIdx) + nIdx] += alpha * a[(k * mIdx) + kIdx] * b[(n * kIdx) + nIdx];
                    
                    }
            }
}

