#pragma once

#include "caffe2/core/tensor.h"
#include "caffe2/utils/Array.h"

namespace caffe2 {
namespace ops {

struct AveragedLoss final {
    struct State final {
        Tensor<CPUContext> scratch;
    };

    static constexpr const char* name = "averaged_loss";

    using Signature = void(const Tensor<CPUContext>& input, Tensor<CPUContext>* output, State* state, CPUContext* context);

    static constexpr c10::guts::array<const char*, 4> parameter_names = {{"input", "output", "state", "context"}};
};

} // namespace ops
} // namespace caffe2
