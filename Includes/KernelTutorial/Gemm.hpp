#include <utility>

using GemmKernel = void (*)(float*, float*, float*, float, float, std::size_t, std::size_t, std::size_t);

void GemmBasic(float* a, float* b, float* c, float alpha, float beta, std::size_t m, std::size_t n, std::size_t k);
void GemmTiling(float* a, float* b, float* c, float alpha, float beta, std::size_t m, std::size_t n, std::size_t k);