#pragma once

#include "caffe2/core/tensor.h"
#include "caffe2/utils/Array.h"

namespace caffe2 {
namespace ops {

struct FullyConnected final {
    static constexpr const char* name = "FC";

    struct Cache final {
        vector<TIndex> Y_shape_cache_;
        Tensor<CPUContext> bias_multiplier_;
    };
    
    using Signature = void(Tensor<CPUContext> X, Tensor<CPUContext> W, Tensor<CPUContext> b, Tensor<CPUContext>* output, Cache* cache, CPUContext* context);

    static constexpr c10::guts::array<const char*, 6> parameter_names = {{"X", "W", "b", "output", "cache", "context"}};
};

} // namespace ops
} // namespace caffe2
