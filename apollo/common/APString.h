#pragma once

#define APOLLO_INLINE inline
#include <assert.h>
#include <stdlib.h>
#include <bit>
#include <cstring>
#include <memory>

namespace apollo
{
    namespace apstring_detail
    {

    };

    
    template<typename T>
    APOLLO_INLINE T* apolloMalloc(size_t size)
    {
        return static_cast<T*>(malloc(size));
    }

    template<typename T>
    APOLLO_INLINE void apolloMalloc(T* data, size_t size)
    {
        data = static_cast<T*>(malloc(size));
    }

    constexpr bool isLittleEndian = std::endian::native == std::endian::little;


    typedef uint8_t apstring_dedicate_type;

    enum class APStringDedicateType : apstring_dedicate_type
    {
        NonDedicate  = 0,
        DedicateSmall = 0x01,
        DedicateMeidum = 0x02,
        DedicateLarge = 0x04,
        DedicateNonLarge = 0x03,
    };

    template<typename Char_Type, 
        APStringDedicateType Dedicate_Type = APStringDedicateType::NonDedicate>
    class apstring_core
    {

        public:
        
            apstring_core() noexcept = default;

            APOLLO_INLINE size_t size() const
            {
                if constexpr(Dedicate_Type == APStringDedicateType::DedicateSmall) 
                {
                    return getSmallSize();
                }
                else if constexpr(Dedicate_Type == APStringDedicateType::DedicateMeidum ||
                                  Dedicate_Type == APStringDedicateType::DedicateLarge)
                {
                    return medString_.size;
                }

                size_t ret = medString_.size;
                if constexpr (isLittleEndian)
                {
                    typedef std::make_unsigned_t<Char_Type> UChar_Type;
                    auto maybeSmallSize = static_cast<size_t>(maxSmallStrSize) - 
                         static_cast<size_t>(static_cast<UChar_Type>(bytes_[lastByte]));
                    return (static_cast<ssize_t>(maybeSmallSize) >= 0) ? maybeSmallSize : ret;
                }
                else
                {
                    return (category() == Category::Small) ? getSmallSize() : ret;
                }
                return ret;
            }
                




        public:
            typedef uint8_t category_type;

            enum class Category : category_type
            {
                Small = 0,
                Medium = isLittleEndian ? 0x80 : 0x02,
                Large = isLittleEndian ? 0x40 : 0x01,
            };

            constexpr static category_type categoryExtractMask = isLittleEndian ? 0xC0 : 0x03; 

            Category category() const
            {
                return static_cast<Category>(bytes_[lastByte] & categoryExtractMask);
            }


        private:
        /*
         *  Code For Small String
         */
            APOLLO_INLINE void initSmall(const Char_Type * const _data, const size_t _size) 
            {
                assert(_size <= maxSmallStrSize);
                
                if(_size > 0) [[likely]]
                {
                    assert(_data != nullptr);
                    memcpy(small_, _data, _size);
                }
                setSmallSize(_size); 
            }

            APOLLO_INLINE void copySmall(const apstring_core& rhs)
            {
                static_assert(offsetof(mediumString, data), "apstring layout failure!");
                static_assert(offsetof(mediumString, size) == sizeof(medString_.data),
                              "apstring layout failure!");
                static_assert(offsetof(mediumString, capacity) == sizeof(medString_.data) + sizeof(medString_.size),
                              "apstring layout failure!");
                medString_ = rhs.medString_;
                assert(category() == Category::Small && size() == _size);
            }

            APOLLO_INLINE void setSmallSize(size_t _size)
            {
                assert(size <= maxSmallStrSize);
                constexpr auto shift = isLittleEndian ? 0 : 2;
                small_[maxSmallStrSize] = static_cast<Char_Type>((maxSmallStrSize - _size) << shift);
                small_[_size] = '\0'; 
                assert(category() == Category::Small && size() == _size);
            }

            APOLLO_INLINE size_t getSmallSize() const
            {
                assert(category() == Category::Small);
                constexpr auto shift = isLittleEndian ? 0 : 2;
                size_t _size = static_cast<size_t>(bytes_[lastByte] >> shift);
                assert(maxSmallStrSize >= _size);
                return maxSmallStrSize - _size;
            }

            void reserveSmall(size_t minCapacity) 
            {

                if(minCapacity <= maxSmallStrSize) [[unlikely]]
                {
                    // Do nothing
                }
                else if(minCapacity <= maxMediumStrSize)
                {
                    // Allocate one more for null terminator
                    const auto allocSize = (minCapacity + 1) * sizeof(Char_Type);
                    auto pData = apolloMalloc<Char_Type>(allocSize);
                    auto _size = getSmallSize();
                    memcpy(pData, small_, _size + 1);

                    medString_.data = pData;
                    medString_.size = _size;
                    medString_.setCapacity(minCapacity, Category::Medium);
                }
                else 
                {
                    // To be done

                }
                
            }



        /*
         *  Code for Medium String
         */
            APOLLO_INLINE void initMedium(const Char_Type * const _data, size_t _size)
            {
                /*
                 *  Some Asserts
                 */
                apolloMalloc(medString_.data, (_size + 1) * sizeof(Char_Type));
                // C++ 20 feature
                if(_size > 0) [[likely]]
                {
                    memcpy(medString_.data, _data, _size);
                }
                medString_.size = _size;
                medString_.setCapacity(_size, Category::Medium);
                medString_.data[_size] = '\0';
            }

            APOLLO_INLINE void copyMedium(const apstring_core& rhs)
            {
                auto allocSize = (medString_.capacity + 1) * sizeof(Char_Type);
                medString_.data = static_cast<Char_Type*>(allocSize);

                assert(medString_.data != nullptr);
                assert(rhs.medString_.data != nullptr);
                assert(medString_ + rhs.medString_.capacity + 1 <= rhs.medString_.data 
                       || medString_.data >= rhs.medString_.data);
                
                memcpy(medString_.data, rhs.medString_.data, sizeof(Char_Type) * (rhs.medString_.size + 1));
                medString_.size = rhs.medString_.size;
                medString_.setCapacity(rhs.medString_.capacity, Category::Medium);
            }

        
        /*
         *   Code for Large String
         */
        APOLLO_INLINE void initLarge(const Char_Type * const _data, size_t _size)
        {


        }

        

        private:

            struct mediumString
            {
                Char_Type* data;
                size_t size;
                size_t capacity;

                APOLLO_INLINE size_t getCapacity() const 
                {
                    return isLittleEndian ? capacity & capacityExtractMask : capacity >> 2;
                }

                void setCapacity(size_t cap, Category cat)
                {
                    capacity = isLittleEndian ?
                        cap | (static_cast<size_t>(cat) << categoryShift) :
                        (cap << 2) | (static_cast<size_t>(cat));
                }
            };

            constexpr static size_t lastByte = sizeof(mediumString) - 1;
            constexpr static size_t maxSmallStrSize = lastByte / sizeof(Char_Type);
            constexpr static size_t categoryShift = (sizeof(size_t) - 1) * 8;
            constexpr static size_t capacityExtractMask = isLittleEndian ?  
                ~(static_cast<size_t>(categoryExtractMask) << categoryShift) : 0x0;
            constexpr static size_t maxMediumStrSize = 254 / sizeof(Char_Type);

            union {
                uint8_t bytes_[sizeof(mediumString)];
                Char_Type small_[maxSmallStrSize];
                mediumString medString_;
            };

            struct RefCounter
            {
                std::atomic_size_t refCount;
                Char_Type data[1];

                constexpr static size_t getDataOffset() 
                {
                    return offsetof(RefCounter, data);
                }
                static RefCounter* create(const Char_Type * const _data, size_t _size)
                {
                    const auto allocSize = (_size + 1) * sizeof(Char_Type) + getDataOffset(); 
                    auto pCounter = apolloMalloc<RefCounter*>(allocSize);
                    pCounter->refCount.store(1, std::memory_order_release);
                    if(_size > 0) [[likely]]
                    {
                        memcpy(pCounter->data, _data, _size);
                    }
                    pCounter->data[_size] = '\0';
                    return pCounter;
                }

                APOLLO_INLINE RefCounter* fromData(Char_Type* p)
                {
                    return static_cast<RefCounter*>(static_cast<std::byte*>(p) - getDataOffset());
                }

                APOLLO_INLINE void increment(Char_Type* p)
                {
                    fromData(p)->refCount.fetch_add(1, std::memory_order_acq_rel);
                }

                APOLLO_INLINE void decrement(Char_Type* p)
                {
                    fromData(p)->refCount.fetch_sub(1, std::memory_order_acq_rel);
                }
            };



    };

};
