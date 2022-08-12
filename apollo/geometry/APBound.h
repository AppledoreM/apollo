#pragma once

#include <concepts>
#include <type_traits>
#include "../math/APVec.h"

namespace apollo
{
    namespace geometry
    {
        enum class APBoundingBoxType
        {
            AABB
        };

        struct APObjectBoundAdapter
        {

        };


        template<APBoundingBoxType BoundType, typename DataType> 
        class APBoundingBox;

        template<typename DataType>
        class APBoundingBox<APBoundingBoxType::AABB, DataType>
        {
        public:
            APBoundingBox(APObjectBoundAdapter* obj) : _obj(std::move(obj))
            {

            }

            APBoundingBox(APBoundingBox&& oth) : _obj(std::move(oth._obj))
            {

            }

            APBoundingBox(APBoundingBox&) = delete;
            APBoundingBox& operator=(APBoundingBox&) = delete;



        private:
            // TODO: add type detection for necessary function
            void calculateGeometryBoundingBox();
            void calculateMeshBoundingBox();

            math::APVector<3, DataType> _bottomLeft, _topRight;
            APObjectBoundAdapter* _obj = nullptr;
        };

    }
}
