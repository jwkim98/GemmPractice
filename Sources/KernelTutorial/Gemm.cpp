#include <KernelTutorial/Gemm.hpp>

#include <intrin.h>
#include <algorithm>

// Basic GEMM
void GemmBasic(float* a, float* b, float* c, float alpha,
               float beta, std::size_t m, std::size_t n,
               std::size_t k)
{
    for (std::size_t mIdx = 0; mIdx < m; ++mIdx)
        for (std::size_t nIdx = 0; nIdx < n; ++nIdx)
        {
            float sum = 0;

            for (std::size_t kIdx = 0; kIdx < k; ++kIdx)
            {
                sum += a[(k * mIdx) + kIdx] * b[(n * kIdx) + nIdx];
            }

            c[(n * mIdx) + nIdx] =
                (alpha * sum) + (beta * c[(n * mIdx) + nIdx]);
        }
}

// GEMM with loop tiling optimization
void GemmTiling(float* a, float* b, float* c, float alpha,
                float beta, std::size_t m, std::size_t n,
                std::size_t k)
{
    for (std::size_t i = 0; i < (m * n); ++i)
        c[i] *= beta;  // beta*C 부분을 미리 계산.

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
                        const float alpha_a = alpha * a[(k * mIdx) + kIdx];

                        for (std::size_t nIdx = nBlock; nIdx < nEnd; ++nIdx)
                            c[(n * mIdx) + nIdx] +=
                                alpha_a * b[(n * kIdx) + nIdx];
                    }
            }
}

void GemmParallel(float* a, float* b, float* c, float alpha,
                  float beta, std::size_t m, std::size_t n,
                  std::size_t k)
{
#pragma omp parallel for collapse(2)
    for (std::size_t mIdx = 0; mIdx < m; ++mIdx)
        for (std::size_t nIdx = 0; nIdx < n; ++nIdx)
            for (std::size_t kIdx = 0; kIdx < k; ++kIdx)
            {
                c[(n * mIdx) + nIdx] +=
                    a[(k * mIdx) + kIdx] * b[(n * kIdx) + nIdx];
            }
}

void GemmParallelTiled(float* a, float* b, float* c, float alpha, float beta,
                       std::size_t m, std::size_t n, std::size_t k)
{
#pragma omp parallel for
    for (std::size_t i = 0; i < (m * n); ++i)
        c[i] *= beta;  // beta*C 부분을 미리 계산.

    constexpr std::size_t CacheSize = 64;

#pragma omp parallel for collapse(2)
    for (std::size_t mBlock = 0; mBlock < m; mBlock += CacheSize)
        for (std::size_t nBlock = 0; nBlock < n; nBlock += CacheSize)
            for (std::size_t kBlock = 0; kBlock < k; kBlock += CacheSize)
            {
                const std::size_t mEnd = std::min(mBlock + CacheSize, m);
                const std::size_t nEnd = std::min(nBlock + CacheSize, n);
                const std::size_t kEnd = std::min(kBlock + CacheSize, k);

                for (std::size_t mIdx = mBlock; mIdx < mEnd; ++mIdx)
                    for (std::size_t kIdx = kBlock; kIdx < kEnd; ++kIdx)
                        for (std::size_t nIdx = nBlock; nIdx < nEnd; ++nIdx)
                            c[(n * mIdx) + nIdx] += alpha *
                                                    a[(k * mIdx) + kIdx] *
                                                    b[(n * kIdx) + nIdx];
            }
}

void GemmUnrolled(float* a, float* b, float* c, float alpha,
                  float beta, std::size_t m, std::size_t n,
                  std::size_t k)
{
    for (std::size_t i = 0; i < (m * n); ++i)
        c[i] *= beta;  // beta*C 부분을 미리 계산.

    constexpr std::size_t CacheSize = 64;
    constexpr std::size_t UnrollBlockSize = 8;

    for (std::size_t mBlock = 0; mBlock < m; mBlock += CacheSize)
        for (std::size_t kBlock = 0; kBlock < k; kBlock += CacheSize)
            for (std::size_t nBlock = 0; nBlock < n; nBlock += CacheSize)
            {
                const std::size_t mEnd =
                    mBlock + CacheSize;  // std::min(mBlock + CacheSize, m);
                // const std::size_t nEnd = nBlock + CacheSize; //
                // std::min(nBlock + CacheSize, n);
                const std::size_t kEnd =
                    kBlock + CacheSize;  // std::min(kBlock + CacheSize, k);

                for (std::size_t mIdx = mBlock; mIdx < mEnd; ++mIdx)
                    for (std::size_t kIdx = kBlock; kIdx < kEnd; ++kIdx)
                    {
                        // const auto loopCount = (nEnd - nBlock) /
                        // UnrollBlockSize; const auto loopRemains = (nEnd -
                        // nBlock) % UnrollBlockSize;
                        const auto loopCount = CacheSize / UnrollBlockSize;
                        const auto alphaA = alpha * a[(k * mIdx) + kIdx];

                        auto bIdx = (n * kIdx) + nBlock;
                        auto cIdx = (n * mIdx) + nBlock;

                        for (std::size_t scaledN = 0; scaledN < loopCount;
                             ++scaledN)
                        {
                            c[cIdx + 0] += alphaA * b[bIdx + 0];
                            c[cIdx + 1] += alphaA * b[bIdx + 1];
                            c[cIdx + 2] += alphaA * b[bIdx + 2];
                            c[cIdx + 3] += alphaA * b[bIdx + 3];
                            c[cIdx + 4] += alphaA * b[bIdx + 4];
                            c[cIdx + 5] += alphaA * b[bIdx + 5];
                            c[cIdx + 6] += alphaA * b[bIdx + 6];
                            c[cIdx + 7] += alphaA * b[bIdx + 7];

                            bIdx += UnrollBlockSize;
                            cIdx += UnrollBlockSize;
                        }

                        // for (auto i = 0; i < loopRemains; ++i)
                        //     c[cIdx++] += alphaA * b[bIdx++];
                    }
            }
}

void GemmVectorized(float* a, float* b, float* c, float alpha,
                    float beta, std::size_t m, std::size_t n,
                    std::size_t k)
{
    for (std::size_t i = 0; i < (m * n); ++i)
        c[i] *= beta;  // beta*C 부분을 미리 계산.

    constexpr std::size_t CacheSize = 64;
    constexpr std::size_t UnrollBlockSize = 8;

    for (std::size_t mBlock = 0; mBlock < m; mBlock += CacheSize)
        for (std::size_t kBlock = 0; kBlock < k; kBlock += CacheSize)
            for (std::size_t nBlock = 0; nBlock < n; nBlock += CacheSize)
            {
                const std::size_t mEnd =
                    mBlock + CacheSize;  // std::min(mBlock + CacheSize, m);
                // const std::size_t nEnd = nBlock + CacheSize; //
                // std::min(nBlock + CacheSize, n);
                const std::size_t kEnd =
                    kBlock + CacheSize;  // std::min(kBlock + CacheSize, k);

                for (std::size_t mIdx = mBlock; mIdx < mEnd; ++mIdx)
                    for (std::size_t kIdx = kBlock; kIdx < kEnd; ++kIdx)
                    {
                        // const auto loopCount = (nEnd - nBlock) /
                        // UnrollBlockSize; const auto loopRemains = (nEnd -
                        // nBlock) % UnrollBlockSize;
                        const auto loopCount = CacheSize / UnrollBlockSize;
                        const auto alphaA = alpha * a[(k * mIdx) + kIdx];
                        const __m256 A = _mm256_broadcast_ss(&alphaA);

                        auto bIdx = (n * kIdx) + nBlock;
                        auto cIdx = (n * mIdx) + nBlock;

                        for (std::size_t scaledN = 0; scaledN < loopCount;
                             ++scaledN)
                        {
                            __m256 C = _mm256_load_ps(&(c[cIdx]));
                            __m256 B = _mm256_load_ps(&(b[bIdx]));
                             C = _mm256_add_ps(C, _mm256_mul_ps(A, B));

                            _mm256_store_ps(&(c[cIdx]), C);

                            bIdx += UnrollBlockSize;
                            cIdx += UnrollBlockSize;
                        }

                        // for (auto i = 0; i < loopRemains; ++i)
                        //     c[cIdx++] += alphaA * b[bIdx++];
                    }
            }
}