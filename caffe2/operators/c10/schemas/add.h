#pragma once

#include "caffe2/core/tensor.h"
#include "caffe2/utils/Array.h"

namespace caffe2 {
namespace ops {

struct Add final {
    static constexpr const char* name = "add";

    using Signature = void(Tensor<CPUContext> input1, Tensor<CPUContext> input2, Tensor<CPUContext>* output, bool legacy_broadcast, int axis, CPUContext* context);

    static constexpr c10::guts::array<const char*, 6> parameter_names = {{"input1", "input2", "output", "legacy_broadcast", "axis", "context"}};
};

} // namespace ops
} // namespace caffe2
