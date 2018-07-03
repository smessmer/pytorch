#include "caffe2/operators/c10/schemas/stop_gradient.h"
#include "caffe2/core/dispatch/KernelRegistration.h"
#include "caffe2/utils/math.h"

using caffe2::Tensor;
using caffe2::CPUContext;

namespace {
template<class DataType>
void stop_gradient_op_cpu_impl(const Tensor<CPUContext>& input, Tensor<CPUContext> *output, CPUContext* context) {
    if (output != &input) {
        output->CopyFrom(input, context);
    }
}
} // namespace

namespace c10 {
    C10_REGISTER_KERNEL(caffe2::ops::StopGradient)
        .kernel(&stop_gradient_op_cpu_impl<float>)
        .dispatchKey({DeviceTypeId::CPU, LayoutId(0), caffe2::TypeMeta::Id<float>()});
} // namespace c10
