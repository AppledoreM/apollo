#pragma once
#include <cstdlib>
#include <stdlib.h>
#include <vector>
#include <utility>
#include <stdio.h>
#include <iostream>
#include <chrono>
#include <algorithm>


#define ERROR_CHECK(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}
namespace apollo
{
    namespace algorithm
    {
        namespace detail
        {
            template<size_t partitionSize, typename T>
            void kernelWrapper(T* arr, std::vector<T*>& blockSumLevels, size_t size, const std::vector<size_t>& numBlocksPerLevel);
        }

        template<typename T>
        class APGPrefixSum
        {
        public:
            typedef T DataType;

            template<size_t partitionSize>
            void prefixSum(DataType* inData, DataType* outData, size_t size)
            {

                uint32_t maxLevels = std::ceil(std::log(size) / std::log(partitionSize));

                if(_numLevels < maxLevels)
                {
                    _numLevels = maxLevels;
                    _blockSumLevels.resize(_numLevels, nullptr);
                    _numBlocksPerLevel.resize(_numLevels, 0);

                    size_t _levelSize = size;

                    for(uint32_t i = 0; i < _numLevels; ++i)
                    {
                        size_t numBlocks = _levelSize / partitionSize + (_levelSize % partitionSize != 0);
                        auto& _blockSumPtr = _blockSumLevels[i];

                        if(_numBlocksPerLevel[i] < numBlocks)
                        {
                            _numBlocksPerLevel[i] = numBlocks;
                            if(_blockSumPtr) cudaFree(_blockSumPtr);
                            cudaMalloc(&_blockSumPtr, numBlocks * sizeof(DataType));
                        }

                        _levelSize = numBlocks;
                    }
                }

                size_t arrSize = size * sizeof(DataType);
                if(_arrLength < size)
                {
                    _arrLength = size;
                    arrSize = _arrLength * sizeof(DataType);
                    if(_arr) cudaFree(_arr);
                    cudaMalloc(&_arr, arrSize); 
                }

                cudaMemcpy(_arr, inData, arrSize, cudaMemcpyHostToDevice);

                auto start = std::chrono::steady_clock::now();
                detail::kernelWrapper<1024>(_arr, _blockSumLevels, size, _numBlocksPerLevel);
                auto end = std::chrono::steady_clock::now();
                //std::cout << "Pure GPU Used time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms\n";

                cudaMemcpy(outData, _arr, arrSize, cudaMemcpyDeviceToHost);
            }

            ~APGPrefixSum()
            {
                std::for_each(std::begin(_blockSumLevels), std::end(_blockSumLevels), [](auto& ptr){ cudaFree(ptr); ptr = nullptr;});
            }


        private:
            uint32_t _numLevels = 0;
            std::vector<size_t> _numBlocksPerLevel;
            std::vector<DataType*> _blockSumLevels;
            DataType *_arr = nullptr;
            size_t _arrLength = 0;

        };

    }
}


