#include <KernelTutorial/Gemm.hpp>

void Gemm(float* a, float* b, float* c, float alpha, float beta, std::size_t m, std::size_t n, std::size_t k)
{
    for(std::size_t mIdx = 0; mIdx < m; ++mIdx)
        for(std::size_t nIdx = 0; nIdx < n; ++nIdx)
            for(std::size_t kIdx = 0; kIdx < k; ++kIdx)
            {
                //! implement here!
            }
}