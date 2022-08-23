#include <utility>

using GemmKernel = void (*)(float* a, float* b, float* c, float alpha,
                            float beta, std::size_t m,
                            std::size_t n, std::size_t k);

void GemmBasic(float* a, float* b, float* c, float alpha,
               float beta, std::size_t m, std::size_t n,
               std::size_t k);
void GemmTiling(float* a, float* b, float* c, float alpha,
                float beta, std::size_t m, std::size_t n,
                std::size_t k);
void GemmParallel(float* a, float* b, float* c, float alpha,
                  float beta, std::size_t m, std::size_t n,
                  std::size_t k);
void GemmParallelTiled(float* a, float* b, float* c, float alpha,
                       float beta, std::size_t m,
                       std::size_t n, std::size_t k);
void GemmUnrolled(float* a, float* b, float* c, float alpha,
                  float beta, std::size_t m, std::size_t n,
                  std::size_t k);
void GemmVectorized(float* a, float* b, float* c, float alpha,
                    float beta, std::size_t m, std::size_t n,
                    std::size_t k);