#include "APGAlgorithm.cuh"
#include <stdio.h>
#include <iostream>
#include <random>
#include <chrono>

namespace apollo
{
    namespace algorithm
    {

        namespace kernel
        {
            template<size_t partitionSize, typename T>
            __global__ void prefixSum(T* inData, T* blockSum, size_t size)
            {
                size_t offset = blockIdx.x * blockDim.x;
                size_t tid = threadIdx.x;

                __shared__ T data[partitionSize + 1];
                if(tid + offset < size)
                {
                    data[tid] = inData[tid + offset];
                }

                size_t pd = 1;
                for(size_t level = partitionSize >> 1; level > 0; level >>= 1, pd <<= 1)
                {
                    __syncthreads();
                    if(tid < level)
                    {
                        size_t dest = (tid + 1) * 2 * pd - 1; 
                        size_t src = dest - pd ;
                        data[dest] += data[src];
                    }
                }

                if(!tid) data[partitionSize - 1] = 0;

                for(size_t d = 1; d < partitionSize; d <<= 1)
                {
                    pd >>= 1;
                    __syncthreads();
                    if(tid < d)
                    {
                        size_t dest = (tid + 1) * 2 * pd - 1; 
                        size_t src = dest - pd;
                        T tmp = data[src];
                        data[src] = data[dest];
                        data[dest] += tmp;
                    }
                }
                __syncthreads();
                if(tid + 1 == partitionSize && tid + offset < size) data[partitionSize] = data[tid] + inData[tid + offset]; 
                if(tid + offset < size) inData[tid + offset] = data[tid + 1];
                __syncthreads();

                if(!tid) blockSum[blockIdx.x] = data[partitionSize];
            }

            template<typename T>
            __global__ void preprocessPrefixSumBlockSum(T* blockSum, size_t numBlocks)
            {

                size_t tid = threadIdx.x + blockIdx.x * blockDim.x;

                if(tid < numBlocks)
                {
                    T temp = blockSum[tid];
                    __syncthreads();

                    for(size_t i = tid + 1; i < numBlocks; ++i)
                    {
                        atomicAdd(&blockSum[i], temp);
                    }
                }

            }

            template<typename T>
            __global__ void distributPrefixBlockSum(T* inData, T* blockSum, size_t size)
            {
                __shared__ T _sum;
                size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
                if(!threadIdx.x)
                {
                    if(blockIdx.x) _sum = blockSum[blockIdx.x - 1];
                    else _sum = 0;
                }
                __syncthreads();


                if(tid < size) inData[tid] += _sum;
            }
        }

        namespace detail
        {

            template<size_t partitionSize, typename T>
            void kernelWrapper(T* arr, T* blockSum, T* temp0, T* temp1, size_t size, size_t numBlocks)
            {

                apollo::algorithm::kernel::prefixSum<partitionSize> <<<numBlocks, 1024>>>(arr, blockSum, size);
                cudaDeviceSynchronize();

                size_t numGrids0 = numBlocks / 1024 + (numBlocks % 1024 != 0);
                /* apollo::algorithm::kernel::preprocessPrefixSumBlockSum<<<numGrids0, 1024>>>(blockSum, numBlocks); */
                /* cudaDeviceSynchronize(); */

                apollo::algorithm::kernel::prefixSum<partitionSize> <<<numGrids0, 1024>>>(blockSum, temp0, numBlocks);
                cudaDeviceSynchronize();

                if(numGrids0 > 1)
                {
                    size_t numGrids1 = numGrids0 / 1024 + (numGrids0 % 1024 != 0);
                    apollo::algorithm::kernel::prefixSum<partitionSize> <<<numGrids1, 1024>>>(temp0, temp1, numGrids0);
                    cudaDeviceSynchronize();
                    apollo::algorithm::kernel::distributPrefixBlockSum<<<numGrids1, 1024>>>(temp0, temp1, numGrids0);
                    cudaDeviceSynchronize();
                }

                apollo::algorithm::kernel::distributPrefixBlockSum<<<numGrids0, 1024>>>(blockSum, temp0, numBlocks);
                cudaDeviceSynchronize();
                apollo::algorithm::kernel::distributPrefixBlockSum<<<numBlocks, 1024>>>(arr, blockSum, size);
                cudaDeviceSynchronize();
            }
        }

    }
}



typedef double DataType;
int main()
{
    constexpr auto arrayLength = 1 << 28;
    auto* arr = new DataType[arrayLength];
    auto* sum = new DataType[arrayLength];

    apollo::algorithm::APGPrefixSum<DataType> ps;

    std::chrono::time_point<std::chrono::steady_clock> start, end;
    for(int epoch = 0; epoch < 100; ++epoch)
    {
        {
            
            std::random_device dev;
            std::mt19937 random_engine(dev());
            std::uniform_int_distribution<std::mt19937::result_type> dist(0, 100);
            for(int i = 0; i < arrayLength; ++i)
            {
                arr[i] = dist(random_engine);
                sum[i] = arr[i];
            }
        }

        bool match = true;

        // CPU Prefix Sum
        {
            start = std::chrono::steady_clock::now();
            for(int i = 1; i < arrayLength; ++i) sum[i] += sum[i - 1];
            end = std::chrono::steady_clock::now();
            std::cout << "CPU Add Finished! Used time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms\n";
        }

        // GPU Prefix Sum
        {
            start = std::chrono::steady_clock::now();
            ps.prefixSum<1024>(arr, arr, arrayLength);
            end = std::chrono::steady_clock::now();
            std::cout << "GPU Add Finished! Used time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms\n";
        }

        {
            for(int i = 0; i < arrayLength; ++i)
            {
                if(std::abs(arr[i] - sum[i]) > 0.0001)
                {
                    std::cout << arr[i] << " " << sum[i] << std::endl;
                    match = false;
                    break;
                }
            }
            if(!match) printf("Results mismatch!\n");
            else printf("Results match!\n");
        
        }
    }
}
