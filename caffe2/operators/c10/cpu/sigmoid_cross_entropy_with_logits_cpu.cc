#include "caffe2/operators/c10/schemas/sigmoid_cross_entropy_with_logits.h"
#include "caffe2/core/dispatch/KernelRegistration.h"
#include "caffe2/utils/math.h"

using caffe2::Tensor;
using caffe2::CPUContext;
using caffe2::TIndex;

namespace {
inline float sigmoid_partition(float lgt) {
    // computes log(1 + exp(lgt)) with only exp(x) function when x >= 0
    return lgt * (lgt >= 0) + log(1 + exp(lgt - 2 * lgt * (lgt >= 0)));
}

inline float sigmoid_xent_forward(float lgt, float tgt) {
    return lgt * (tgt - (lgt >= 0)) - log(1 + exp(lgt - 2 * lgt * (lgt >= 0)));
}

inline float sigmoid_xent_forward_with_log_d_trick(float lgt, float tgt) {
    return (2 * tgt - 1.) * (lgt - sigmoid_partition(lgt));
}

inline float unjoined_sigmoid_xent_forward(float lgt, float tgt) {
    return lgt * tgt + (tgt - 1) * lgt * (lgt >= 0) -
           (1 - tgt) * log(1 + exp(lgt - 2 * lgt * (lgt >= 0)));
}

void sigmoid_cross_entropy_with_logits_op_cpu_impl(const Tensor<CPUContext>& logits, const Tensor<CPUContext>& targets, Tensor<CPUContext>* out, bool log_D_trick, bool unjoined_lr_loss) {
    CAFFE_ENFORCE_EQ(logits.dims(), targets.dims());
    const auto inner_size = logits.ndim() > 0 ? logits.dims().back() : 1;
    const auto outer_size = logits.size() / inner_size;

    if (logits.ndim() == 0) {
        out->Resize(std::vector<TIndex>{});
    } else {
        std::vector<TIndex> dims(logits.dims().begin(), logits.dims().end() - 1);
        out->Resize(dims);
    }
    auto* out_ptr = out->mutable_data<float>();

    auto* logits_ptr = logits.data<float>();
    auto* targets_ptr = targets.data<float>();

    auto in_idx = 0;
    for (int i = 0; i < outer_size; ++i) {
        float value = 0;
        for (int j = 0; j < inner_size; ++j) {
            if (unjoined_lr_loss) {
                value += unjoined_sigmoid_xent_forward(
                        logits_ptr[in_idx], targets_ptr[in_idx]);
            } else {
                value +=
                        (log_D_trick ? sigmoid_xent_forward_with_log_d_trick(
                                logits_ptr[in_idx], targets_ptr[in_idx])
                                      : sigmoid_xent_forward(
                                        logits_ptr[in_idx], targets_ptr[in_idx]));
            }
            ++in_idx;
        }
        out_ptr[i] = -value / inner_size;
    }
}
} // namespace

namespace c10 {
    C10_REGISTER_KERNEL(caffe2::ops::SigmoidCrossEntropyWithLogits)
        .kernel(&sigmoid_cross_entropy_with_logits_op_cpu_impl)
        .dispatchKey(c10::DispatchKey<2>{
                c10::details::TensorParameterDispatchKey{DeviceTypeId::CPU, LayoutId(0), caffe2::TypeMeta::Id<float>()},
                c10::details::TensorParameterDispatchKey{DeviceTypeId::CPU, LayoutId(0), caffe2::TypeMeta::Id<float>()}
        });
} // namespace c10
