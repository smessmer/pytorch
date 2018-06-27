#pragma once

#include "caffe2/core/tensor.h"
#include "caffe2/utils/Array.h"

namespace caffe2 {
namespace ops {

struct Add final {
    static constexpr const char* name = "add";

    using Signature = void(Tensor<CPUContext> input1, Tensor<CPUContext> input2, Tensor<CPUContext>* output, CPUContext* context);

    static constexpr c10::guts::array<const char*, 4> parameter_names = {{"input1", "input2", "output", "context"}};
};

} // namespace ops
} // namespace caffe2
