#pragma once

#include "caffe2/core/tensor.h"
#include "caffe2/utils/Array.h"

namespace caffe2 {
namespace ops {

struct SparseLengthsSum final {
  static constexpr const char* name = "sparse_lengths_sum";

  using Signature = void(
      const Tensor<CPUContext>& data,
      const Tensor<CPUContext>& indices,
      const Tensor<CPUContext>& lengths,
      Tensor<CPUContext>* output);

  static constexpr c10::guts::array<const char*, 4> parameter_names = {
      {"data", "indices", "lengths", "output"}};
};

} // namespace ops
} // namespace caffe2
