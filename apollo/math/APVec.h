#pragma once
#include <stdlib.h>
#include <cstring>
#include <utility>>

// TODO: Merge VMathLibrary
namespace apollo
{
     namespace math
     {

         template<size_t N, typename DataType>
         class APVector
         {
         public:
             APVector() : _data{}
             {
             }

             APVector(APVector& oth)
             {
                 constexpr size_t _copySize = N * sizeof(DataType);
                 memcpy(_data, oth, _copySize);
             }

             APVector(APVector&& oth) : _data(std::move(oth._data))
             {

             }

             APVector& operator=(APVector& oth) 
             {
                 constexpr size_t _copySize = N * sizeof(DataType);
                 memcpy(_data, oth, _copySize);
                 return *this;
             }

             APVector operator+(const APVector& oth)
             {
                APVector res;
                for(int i = 0; i < N; ++i) res._data[i] = _data[i] + oth._data[i];
             }

             APVector operator*(const APVector& oth)
             {
                APVector res;
                for(int i = 0; i < N; ++i) res._data[i] = _data[i] * oth._data[i];
             }

             APVector operator/(const APVector& oth)
             {
                APVector res;
                for(int i = 0; i < N; ++i) res._data[i] = _data[i] / oth._data[i];
             }


         private:
             DataType _data[N];
         };


     }
}
