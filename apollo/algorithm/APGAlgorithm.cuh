#pragma once
#include <cstdlib>
#include <stdlib.h>
#include <vector>
#include <utility>
#include <stdio.h>
#include <iostream>
#include <chrono>


namespace apollo
{
    namespace algorithm
    {
        namespace detail
        {
            template<size_t partitionSize, typename T>
            void kernelWrapper(T* arr, T* blockSum, T *temp0, T *temp1, size_t size, size_t numBlocks);
        }

        template<typename T>
        class APGPrefixSum
        {
        public:
            typedef T DataType;
            
            template<size_t partitionSize>
            void prefixSum(DataType* inData, DataType* outData, size_t size)
            {
                // Calculate number of blocks needed;
                size_t numBlocks = size / partitionSize + (size % partitionSize != 0);
                if(_numBlocks < numBlocks) 
                {
                    _numBlocks = numBlocks;
                    if(_blockSumPtr) cudaFree(_blockSumPtr);
                    cudaMalloc(&_blockSumPtr, _numBlocks * sizeof(DataType));
                }

                size_t arrSize = _arrLength * sizeof(DataType);
                if(_arrLength < size)
                {
                    _arrLength = size;
                    arrSize = _arrLength * sizeof(DataType);
                    if(_arr) cudaFree(_arr);
                    cudaMalloc(&_arr, arrSize); 
                }

                if(!temp0) cudaMalloc(&temp0, 1024 * 1024 * sizeof(DataType));
                if(!temp1) cudaMalloc(&temp1, 1024 * sizeof(DataType));

                cudaMemcpy(_arr, inData, arrSize, cudaMemcpyHostToDevice);
                auto start = std::chrono::steady_clock::now();
                detail::kernelWrapper<1024>(_arr, _blockSumPtr, temp0, temp1, size, _numBlocks);
                auto end = std::chrono::steady_clock::now();
                std::cout << "Pure GPU Used time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms\n";
                cudaMemcpy(outData, _arr, arrSize, cudaMemcpyDeviceToHost);
            }

            ~APGPrefixSum()
            {
                if(_blockSumPtr) 
                {
                    cudaFree(_blockSumPtr);
                    _blockSumPtr = nullptr;
                }
                if(_arr)
                {
                    cudaFree(_arr);
                    _arr = nullptr;
                }
            }

        private:
            size_t _numBlocks = 0;
            std::vector<DataType> _blockSum;
            DataType *_blockSumPtr = nullptr;

            size_t _arrLength = 0;
            DataType *_arr = nullptr;

            DataType *temp0 = nullptr;
            DataType *temp1 = nullptr;
        };

    }
}


