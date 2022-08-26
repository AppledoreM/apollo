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
                // TODO: Make this faster
                if(tid + 1 == partitionSize && tid + offset < size) data[partitionSize] = data[tid] + inData[tid + offset]; 
                if(tid + offset < size) inData[tid + offset] = data[tid + 1];
                __syncthreads();

                if(!tid) blockSum[blockIdx.x] = data[partitionSize];
            }


            template<typename T>
            __global__ void distributPrefixBlockSum(T* inData, T* blockSum, size_t size)
            {
                __shared__ T _sum;
                size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
                if(threadIdx.x == 0)
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
            void kernelWrapper(T* arr, std::vector<T*>& blockSumLevels, size_t size, const std::vector<size_t>& numBlocksPerLevel)
            {

                apollo::algorithm::kernel::prefixSum<partitionSize> <<<numBlocksPerLevel[0], 1024>>>(arr, blockSumLevels[0], size);
                cudaDeviceSynchronize();

                for(uint32_t i = 1; i < numBlocksPerLevel.size(); ++i)
                {
                    apollo::algorithm::kernel::prefixSum<partitionSize> 
                        <<<numBlocksPerLevel[i], 1024>>>(blockSumLevels[i - 1], blockSumLevels[i], numBlocksPerLevel[i - 1]);
                    cudaDeviceSynchronize();
                }


                for(uint32_t i = numBlocksPerLevel.size() - 1; i > 0; --i)
                {
                    auto* partialSum = new T[numBlocksPerLevel[i - 1]];
                    printf("Distribute Level %u\n", i);
                    apollo::algorithm::kernel::distributPrefixBlockSum
                        <<<numBlocksPerLevel[i], 1024>>>(blockSumLevels[i - 1], blockSumLevels[i], numBlocksPerLevel[i - 1]);
                    cudaDeviceSynchronize();
                    cudaMemcpy(partialSum, blockSumLevels[i - 1], numBlocksPerLevel[i - 1] * sizeof(T), cudaMemcpyDeviceToHost);
                    printf("Partial Sum Level %u\n", i);
                    for(int j = 0; j < numBlocksPerLevel[i - 1]; ++j)
                    {
                        printf("%f, ", partialSum[j]);
                    }
                    delete[] partialSum;
                    printf("\n");
                }

                apollo::algorithm::kernel::distributPrefixBlockSum<<<numBlocksPerLevel[0], 1024>>>(arr, blockSumLevels[0], size);
                cudaDeviceSynchronize();
            }
        }

    }
}



typedef double DataType;
int main()
{

    apollo::algorithm::APGPrefixSum<DataType> ps;

    std::chrono::time_point<std::chrono::steady_clock> start, end;
    std::random_device dev;
    std::mt19937 random_engine(dev());
    std::uniform_int_distribution<std::mt19937::result_type> dist(0, 1);
    std::uniform_int_distribution<std::mt19937::result_type> sizeDist(1, 1 << 22);

    bool allMatch = true;
    float cpuTime = 0;
    float gpuTime = 0;

    std::vector<DataType> arr, sum, gpu;
    for(int epoch = 0; epoch < 10; ++epoch)
    {
        size_t arrayLength = sizeDist(random_engine);
        arr.resize(arrayLength);
        sum.resize(arrayLength);
        gpu.resize(arrayLength);


        {
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
            cpuTime += std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        }

        // GPU Prefix Sum
        {
            start = std::chrono::steady_clock::now();
            ps.prefixSum<1024>(arr.data(), gpu.data(), arrayLength);
            end = std::chrono::steady_clock::now();
            gpuTime += std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        }

        {
            for(int i = 0; i < arrayLength; ++i)
            {
                if(std::abs(gpu[i] - sum[i]) > 0.0001)
                {

                    printf("----------------ERROR LOG---------------\n");
                    for(int j = i; j <= i; ++j)
                    {
                        std::cout << "Size: " <<  arrayLength << \
                            " Index: " << j << \
                            " Original: " << arr[j] \
                            << " GPU Results: " << gpu[j] << \
                            " CPU Results: " << sum[j] << std::endl;
                    }
                    match = false;
                    break;
                }
            }
            if(!match) 
            {
                allMatch = false;
                printf("Results mismatch!\n");
                break;
            }
        }
    }
    if(allMatch) printf("All Results Match!\n");
    std::cout << "CPU Average Used Time: " << cpuTime / 100 << "ms\n";
    std::cout << "GPU Average Used Time: " << gpuTime / 100 << "ms\n";
}
