#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include <KernelTutorial/Test.hpp>
#include <KernelTutorial/Gemm.hpp>

TEST_CASE("Simple test")
{
    CHECK(Add(2, 3) == 5);
}

TEST_CASE("Gemm")
{
    //! Test your Gemm here!
    
}