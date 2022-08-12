#pragma once

#include <concepts>

namespace apollo
{
    namespace geometry
    {
        enum class APBoundBoxType
        {
            AABB
        };


        template<typename BoundType> 
            requires std::is_same_v<BoundType, APBoundBoxType>
        class APBoundBox
        {

        };

    }
}
