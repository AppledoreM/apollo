#include "APGAlgorithm.cuh"


int main()
{
    constexpr auto arrayLength = 5;
    int arr[arrayLength] = {1, 2, 3, 4, 5};
    apollo::algorithm::APGPrefixSum<int> ps;
    ps.prefixSum<1024>(arr, arr, 5);

}
